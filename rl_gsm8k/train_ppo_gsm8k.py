import os
import re
import json
import time
import random
from dataclasses import dataclass, asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb



def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier_if_ddp(enabled: bool):
    if enabled:
        dist.barrier()


def shard_indices(n: int, rank: int, world: int):
    per = (n + world - 1) // world
    s = rank * per
    e = min(n, s + per)
    return list(range(s, e))


# -----------------------------
# GSM8K 
# -----------------------------
def extract_final_number(text: str):
    """
    GSM8K typical answers often end with '#### 18'.
    1) Prefer the number following ####
    2) Otherwise, take the last number in the text
    """
    if text is None:
        return None

    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1])
        except Exception:
            pass

    m2 = re.findall(r"([-+]?\d+(?:\.\d+)?)", text)
    if m2:
        try:
            return float(m2[-1])
        except Exception:
            return None
    return None


def gsm8k_gold_number(answer_field: str):
    return extract_final_number(answer_field)


# -----------------------------
# logprob 
# -----------------------------
def compute_logprobs(model, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     prompt_lens: torch.Tensor):
    """
    Compute the sum of log probabilities for the response part of each sample.

    input_ids: (B, L)
    attention_mask: (B, L)
    prompt_lens: (B,)  Length of the prompt (including latent tokens) for each sample
    Returns:
      logp_sum: (B,)
      token_count: (B,)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, L, V)

    # The token at position t is predicted by the logits at position t-1
    logp_all = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
    targets = input_ids[:, 1:]  # (B, L-1)

    gathered = torch.gather(logp_all, dim=-1,
                            index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    B, Lm1 = gathered.shape
    logp_sum = torch.zeros((B,), device=input_ids.device, dtype=gathered.dtype)
    tok_cnt = torch.zeros((B,), device=input_ids.device, dtype=torch.long)

    for i in range(B):
        pl = int(prompt_lens[i].item())
        # The logp of the first response token is at position pl-1
        start = max(pl - 1, 0)
        end = Lm1

        mask = attention_mask[i, 1:]        # align gathered
        seg_mask = mask[start:end].to(torch.bool)
        seg = gathered[i, start:end]
        seg = seg[seg_mask]

        logp_sum[i] = seg.sum()
        tok_cnt[i] = seg.numel()

    return logp_sum, tok_cnt


# -----------------------------
# config
# -----------------------------
@dataclass
class TrainCfg:
    model_dir: str = "/workspace/coconut_hf_model"
    train_json: str = "/workspace/gsm8k/train.json"
    test_json: str = "/workspace/gsm8k/test.json"

    k_latent: int = 2         # number of latent tokens at the end of the prompt
    group_size: int = 4        # how many generations per prompt (for GRPO-style advantage)
    max_new_tokens: int = 128

    batch_prompts: int = 8     # how many prompts per step (each multiplied by group_size)
    lr: float = 1e-6

    # PPO hyperparameters
    ppo_epochs: int = 4
    ppo_batch_size: int = 32   # how many samples per PPO mini-batch
    clip_range: float = 0.2
    entropy_coef: float = 0.0  # entropy coefficient, disabled by default

    steps: int = 2000
    log_every: int = 20
    save_every: int = 2000
    out_dir: str = "/workspace/ppo_runs"

    # sampling
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95

    normalize_adv: bool = True

    # eval
    eval_every: int = 20             # how often to run eval (in steps)
    eval_batch_prompts: int = 32      # how many prompts per eval batch
    eval_batches: int = 8             # how many eval batches to run (total eval samples ~ eval_batch_prompts*eval_batches)
    eval_max_new_tokens: int = 128    # max new tokens to generate during eval
    eval_do_sample: bool = False      # use greedy decoding during eval (False)

    # wandb
    wandb_project: str = "coconut-ppo-gsm8k"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"   # online / offline / disabled


# -----------------------------
# Eval
# -----------------------------
def evaluate_on_gsm8k(model, tok, cfg: TrainCfg, device, latent_id: int):
    """
    Only called on rank0: randomly sample eval on test.json, return accuracy.
    """
    model_w = model.module if isinstance(model, DDP) else model

    # Save/restore train/eval state
    was_training = model_w.training
    model_w.eval()

    # Load test data
    data = json.load(open(cfg.test_json, "r"))
    n = len(data)
    if n == 0:
        return 0.0

    correct = 0
    total = 0

    def build_prompt(question: str):
        q = question.strip() + "\n"
        ids = tok.encode(q, add_special_tokens=True)
        ids = ids + [latent_id] * cfg.k_latent
        return ids

    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            if n == 0:
                break
            batch_docs = random.sample(data, k=min(cfg.eval_batch_prompts, n))
            prompt_ids_list = [build_prompt(d["question"]) for d in batch_docs]
            gold_nums = [gsm8k_gold_number(d["answer"]) for d in batch_docs]

            Bp = len(prompt_ids_list)
            if Bp == 0:
                continue

            prompt_lens = torch.tensor(
                [len(x) for x in prompt_ids_list],
                device=device,
                dtype=torch.long,
            )
            max_pl = int(prompt_lens.max().item())

            prompt_batch = torch.full(
                (Bp, max_pl),
                tok.eos_token_id,
                device=device,
                dtype=torch.long,
            )
            prompt_mask = torch.zeros(
                (Bp, max_pl),
                device=device,
                dtype=torch.long,
            )
            for i, ids in enumerate(prompt_ids_list):
                prompt_batch[i, :len(ids)] = torch.tensor(ids, device=device)
                prompt_mask[i, :len(ids)] = 1

            # eval: generate 1 greedy response per prompt
            gen_ids = model_w.generate(
                input_ids=prompt_batch,
                attention_mask=prompt_mask,
                max_new_tokens=cfg.eval_max_new_tokens,
                do_sample=cfg.eval_do_sample,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            for i in range(Bp):
                pred_num = extract_final_number(texts[i])
                gold = gold_nums[i]
                if pred_num is not None and gold is not None and abs(pred_num - gold) < 1e-6:
                    correct += 1
                total += 1

    acc = float(correct) / max(total, 1)

    if was_training:
        model_w.train()

    return acc


# -----------------------------
# Main logic (PPO)
# -----------------------------
def main():
    cfg = TrainCfg()

    ddp, local_rank, rank, world = setup_ddp()
    device = torch.device("cuda", local_rank)

    os.makedirs(cfg.out_dir, exist_ok=True)
    if is_rank0(rank):
        print(f"[rank0] ddp={ddp} world={world} device={device}")

    # wandb only initialized on rank0
    wandb_run = None
    if is_rank0(rank) and cfg.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        run_name = cfg.wandb_run_name or f"coconut_ppo_{int(time.time())}"
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=asdict(cfg),
        )
        print(f"[rank0] wandb run: {wandb_run.name}")

    # tokenizer + model（actor）
    tok = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_dir, trust_remote_code=True
    ).to(device)
    model.train()

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    latent_id = tok.convert_tokens_to_ids("<|latent|>")
    assert latent_id is not None and latent_id >= 0

    # training data
    data = json.load(open(cfg.train_json, "r"))
    idxs = shard_indices(len(data), rank, world)
    shard = [data[i] for i in idxs]
    if is_rank0(rank):
        print(f"[rank0] dataset={len(data)} shard_per_rank~{len(shard)}")

    opt = torch.optim.AdamW((model.module if ddp else model).parameters(), lr=cfg.lr)

    def build_prompt(question: str):
        q = question.strip() + "\n"
        ids = tok.encode(q, add_special_tokens=True)
        ids = ids + [latent_id] * cfg.k_latent
        return ids

    # training loop
    t0 = time.time()
    step = 0
    running = {"reward": 0.0, "kl": 0.0, "loss": 0.0}

    while step < cfg.steps:
        # ====== 1. Generate a batch of samples using the current policy ======
        batch_docs = random.sample(shard, k=min(cfg.batch_prompts, len(shard)))
        prompt_ids_list = [build_prompt(d["question"]) for d in batch_docs]
        gold_nums = [gsm8k_gold_number(d["answer"]) for d in batch_docs]

        Bp = len(prompt_ids_list)
        G = cfg.group_size

        # Build prompt batch and repeat G times
        prompt_lens = torch.tensor(
            [len(x) for x in prompt_ids_list], device=device, dtype=torch.long
        )
        max_pl = int(prompt_lens.max().item())

        prompt_batch = torch.full(
            (Bp, max_pl),
            tok.eos_token_id,
            device=device,
            dtype=torch.long,
        )
        prompt_mask = torch.zeros(
            (Bp, max_pl), device=device, dtype=torch.long
        )
        for i, ids in enumerate(prompt_ids_list):
            prompt_batch[i, : len(ids)] = torch.tensor(ids, device=device)
            prompt_mask[i, : len(ids)] = 1

        prompt_batch_rep = prompt_batch.repeat_interleave(G, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(G, dim=0)
        prompt_lens_exp = prompt_lens.repeat_interleave(G)  # (Bp*G,)

        model_w = model.module if ddp else model

        with torch.no_grad():
            gen_ids = model_w.generate(
                input_ids=prompt_batch_rep,
                attention_mask=prompt_mask_rep,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        B = gen_ids.size(0)
        assert B == Bp * G, f"expected {Bp*G} samples but got {B}"

        attn = (gen_ids != tok.eos_token_id).long()

        # ====== 2. Calculate reward (whether the numerical value is correct) ======
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        rewards = torch.zeros((B,), device=device, dtype=torch.float32)

        for i in range(B):
            pidx = i // G
            pred_num = extract_final_number(texts[i])
            gold = gold_nums[pidx]
            if pred_num is None or gold is None:
                rewards[i] = 0.0
            else:
                rewards[i] = 1.0 if abs(pred_num - gold) < 1e-6 else 0.0

        # GRPO-style: within-group mean-subtracted advantage 
        # not standard PPO, but works better for GSM8K
        r_group = rewards.view(Bp, G)
        adv = r_group - r_group.mean(dim=1, keepdim=True)
        if cfg.normalize_adv:
            std = r_group.std(dim=1, keepdim=True) + 1e-6
            adv = adv / std
        adv = adv.view(B)
        adv_detach = adv.detach()

        reward_mean = rewards.mean().item()

        # ====== 3. Calculate old logp (no gradient) ======
        with torch.no_grad():
            logp_old, tokcnt = compute_logprobs(
                model_w, gen_ids, attn, prompt_lens_exp
            )
        tokcnt_f = tokcnt.clamp(min=1).to(torch.float32)
        logp_old_norm = (logp_old / tokcnt_f).detach()

        # ====== 4. PPO multi-epoch update ======
        # We reuse this batch of samples for multiple epochs, PPO clipped objective
        B_indices = torch.arange(B, device=device)
        kl_accum = 0.0
        kl_count = 0
        last_loss_pg = 0.0

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=device)
            B_indices = B_indices[perm]

            for start in range(0, B, cfg.ppo_batch_size):
                end = min(start + cfg.ppo_batch_size, B)
                if start >= end:
                    continue
                idx_mb = B_indices[start:end]

                input_mb = gen_ids[idx_mb]
                attn_mb = attn[idx_mb]
                pl_mb = prompt_lens_exp[idx_mb]
                adv_mb = adv_detach[idx_mb]
                logp_old_mb = logp_old_norm[idx_mb]
                tokcnt_mb = tokcnt_f[idx_mb]

                opt.zero_grad(set_to_none=True)

                logp_new, _ = compute_logprobs(
                    model_w, input_mb, attn_mb, pl_mb
                )
                logp_new_norm = logp_new / tokcnt_mb

                # PPO ratio
                ratio = torch.exp(logp_new_norm - logp_old_mb)
                # clipped surrogate
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(
                    ratio,
                    1.0 - cfg.clip_range,
                    1.0 + cfg.clip_range,
                ) * adv_mb
                loss_pg = -torch.mean(torch.min(surr1, surr2))

                # Approximate KL (old - new)
                kl_batch = (logp_old_mb - logp_new_norm).mean()
                kl_accum += kl_batch.item()
                kl_count += 1
                last_loss_pg = loss_pg.item()

                loss = loss_pg  # If adding entropy, add it here
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_w.parameters(), 1.0)
                opt.step()

        step += 1
        avg_kl = kl_accum / max(kl_count, 1)
        running["reward"] += reward_mean
        running["kl"] += avg_kl
        running["loss"] += last_loss_pg

        # ====== 5. Training log ======
        if step % cfg.log_every == 0 and is_rank0(rank):
            avg_loss = running["loss"] / cfg.log_every
            avg_rew = running["reward"] / cfg.log_every
            avg_kl2 = running["kl"] / cfg.log_every
            dt = time.time() - t0
            print(
                f"[step {step}] "
                f"loss={avg_loss:.4f} "
                f"reward={avg_rew:.3f} "
                f"kl={avg_kl2:.4f} "
                f"dt={dt:.1f}s"
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/reward": avg_rew,
                        "train/kl": avg_kl2,
                        "train/steps_per_sec": cfg.log_every / max(dt, 1e-6),
                        "train/step": step,
                    },
                    step=step,
                )

                sample_text = texts[0] if len(texts) > 0 else ""
                wandb.log({"examples/sample_output": sample_text}, step=step)

            running = {"reward": 0.0, "kl": 0.0, "loss": 0.0}
            t0 = time.time()

        # ====== 6. Eval ======
        if step % cfg.eval_every == 0 and is_rank0(rank):
            eval_acc = evaluate_on_gsm8k(
                model,
                tok,
                cfg,
                device,
                latent_id,
            )
            print(f"[step {step}] eval accuracy on test.json = {eval_acc:.4f}")
            if wandb_run is not None:
                wandb.log(
                    {"eval/accuracy": eval_acc, "train/step": step},
                    step=step,
                )

        # ====== 7. Save ======
        if step % cfg.save_every == 0 and is_rank0(rank):
            save_path = os.path.join(cfg.out_dir, f"step_{step}")
            os.makedirs(save_path, exist_ok=True)
            to_save = model.module if ddp else model
            to_save.save_pretrained(save_path)
            tok.save_pretrained(save_path)
            print(f"[rank0] saved to {save_path}")

    barrier_if_ddp(ddp)
    if is_rank0(rank):
        print("done")

    if wandb_run is not None and is_rank0(rank):
        wandb.finish()


if __name__ == "__main__":
    main()
