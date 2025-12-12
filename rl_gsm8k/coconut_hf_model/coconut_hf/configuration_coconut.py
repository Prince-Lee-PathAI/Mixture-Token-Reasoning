
from transformers import PretrainedConfig

class CoconutConfig(PretrainedConfig):
    model_type = "coconut_no_shared"

    def __init__(
        self,
        base_model_config=None,
        latent_token_id=None,
        start_latent_id=None,
        end_latent_id=None,
        eos_token_id=None,
        state_combination_method="cross_attention",
        combination_use_gating=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_config = base_model_config or {}
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.state_combination_method = state_combination_method
        self.combination_use_gating = combination_use_gating
