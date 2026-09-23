"""Day 13: B4 -- ACT/Bi-ACT trained from scratch, identical data and output
space as B5 (proposal §6.1 baseline table): "ACT/Bi-ACT from scratch, same
bilateral data, same compliance output -- isolates: H4, the pretraining
contribution." B4 is the control H4 needs: if B5 (pretrained VLM) and B4
(no pretraining) degrade identically on held-out language cells (E3/E4),
the vision-language pretraining isn't actually buying anything and H4 is
falsified.

Reuses lerobot's own ACT implementation (`lerobot.policies.act`, the
"Bi-ACT" family this proposal cites descends from ACT/ACT+ variants)
unmodified -- no changes to `modeling_act.py`. Everything below is a thin
wrapper that:
  1. Turns off ImageNet pretraining on the ResNet vision backbone
     (`pretrained_backbone_weights=None`) -- ACTConfig's own default DOES
     load ImageNet weights, which would silently smuggle a form of
     pretraining into a baseline whose entire point is having none. This is
     the one line in this module most likely to be gotten wrong by copying
     ACTConfig's defaults, so it is called out here and asserted in
     `build_bi_act_policy` and the self-test below.
  2. Adds a language pathway "from scratch": the SAME tokenizer vocabulary
     B0/B2/B5 use (config.tokenizer_name, default matches
     ComplianceSmolVLAConfig's vlm_model_name) -- a tokenizer is an integer
     encoding scheme, not learned semantics, so reusing it doesn't leak
     pretraining -- but a freshly Xavier-initialized nn.Embedding instead of
     loading SmolVLM2's own pretrained token embeddings. See
     FromScratchLanguageEncoder.
  3. Adds the same force-history token B2/B5 get (compliance_vla.policy.force_encoder.
     ForceHistoryEncoder, reused unmodified, same augmentations), fed to ACT
     as (part of) its `observation.environment_state` input -- ACT's own
     conditioning slot for a flat non-image, non-robot-state vector (see
     `env_state_feature` in modeling_act.ACT.forward), chosen over modifying
     ACT's transformer internals to keep this baseline architecturally
     boring, matching proposal §4.2's "keep the architecture boring on
     purpose" applied to B4 as much as B5.
  4. Outputs the same 13-dim [x_eq(6), log_k(6), gripper(1)] action ACT
     already regresses directly (no flow-matching -- ACT decodes actions in
     one forward pass), and reuses compliance_vla.policy.losses.compliance_loss for the
     masked-Huber-on-log_k / masked-L1-on-x_eq loss. compliance_loss's
     `u_t, v_t` naming is a flow-matching artifact (residual = u_t - v_t);
     the function body is agnostic to *why* the two tensors differ, so it
     is called here with (target, prediction) instead of (noise, velocity)
     -- reused, not duplicated.

Bilateral data: like B2/B5, B4 trains on `observation.state` = (q, q_dot,
x_f) -- the follower's own joint state and Cartesian pose -- which already
encodes bilateral-teleop-derived quantities (x_f comes from FK, calibrated
against the bilateral rig, src/compliance_vla/policy/labels.py); the leader's pose x_l only
ever appears as the x_eq *target*, for all baselines, per the proposal's
identifiability argument (§2.2b) -- B4 does not get a separate leader-state
input channel beyond what B2/B5 already have, since "same data" (§6.1) means
matching the input space, not adding new information no other baseline sees.
"""

from dataclasses import dataclass

import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .force_encoder import ForceHistoryEncoder, force_dropout, wrench_bias_injection
from .losses import compliance_loss

FORCE_EMBED_DIM = 64
LANG_EMBED_DIM = 64
ENV_STATE_DIM = FORCE_EMBED_DIM + LANG_EMBED_DIM  # scripts/train_policy.py's b4 input_features needs this


class FromScratchLanguageEncoder(nn.Module):
    """Bag-of-embeddings language encoder, trained from scratch (Day 13,
    B4/H4 control). Mean-pools token embeddings over non-padding positions
    (order-invariant -- deliberately no positional encoding or attention, to
    keep this pathway as architecturally simple as the rest of B4) then
    projects to `out_dim`. The embedding table is Xavier-initialized and has
    no relationship to SmolVLM2's own token embeddings beyond sharing the
    same integer vocabulary -- this is what makes it a genuine from-scratch
    language pathway rather than a smaller pretrained one.
    """

    def __init__(self, vocab_size, embed_dim=LANG_EMBED_DIM, out_dim=LANG_EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, token_ids, attention_mask):
        """token_ids: (B, L) int64. attention_mask: (B, L) bool, True=real token."""
        emb = self.embed(token_ids)  # (B, L, E)
        mask = attention_mask.to(emb.dtype).unsqueeze(-1)  # (B, L, 1)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)  # avoid div-by-zero on an all-pad row (shouldn't occur, but never NaN)
        pooled = summed / count
        return self.out_proj(pooled)


@PreTrainedConfig.register_subclass("act_compliance")
@dataclass
class BiActComplianceConfig(ACTConfig):
    # H = 32 @ 30Hz, matching B0/B2/B5 (proposal §4.2) -- overrides ACTConfig's Aloha-tuned 100/100 defaults.
    chunk_size: int = 32
    n_action_steps: int = 32

    # THE from-scratch control's load-bearing line: ACTConfig's own default is
    # "ResNet18_Weights.IMAGENET1K_V1". None here is not an oversight.
    pretrained_backbone_weights: str | None = None

    # Force-history token (§4.2), identical machinery/augmentations to B2/B5.
    force_history_len: int = 20
    force_history_window_sec: float = 0.5
    force_dropout_p: float = 0.15
    wrench_bias_range_n: float = 1.0
    force_embed_dim: int = FORCE_EMBED_DIM

    # Language pathway (from scratch, see FromScratchLanguageEncoder).
    # tokenizer_name only selects a *tokenization scheme* (an integer encoding,
    # not learned weights) -- matches ComplianceSmolVLAConfig's vlm_model_name
    # so B4 sees instructions split into the same subword units B0/B2/B5 do.
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    lang_embed_dim: int = LANG_EMBED_DIM
    lang_vocab_size: int = 0  # placeholder; build_bi_act_policy() fills this in from the real tokenizer before construction

    # Compliance-head loss (§4.2): masked Huber on log_k, weight tuned on val -- same knobs as B5's config.
    huber_delta: float = 1.0
    lam_log_k: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.normalization_mapping.setdefault("ENV", self.normalization_mapping.get("STATE"))


class BiActCompliancePolicy(ACTPolicy):
    """B4: lerobot's unmodified ACT model (config.use_vae default True, i.e.
    genuinely the CVAE-based "ACT" the proposal's "ACT/Bi-ACT" naming refers
    to), plus a force-history + from-scratch-language `environment_state`
    input, plus the 13-dim compliance output/loss B2/B5 already use.
    """

    config_class = BiActComplianceConfig
    name = "act_compliance"

    def __init__(self, config: BiActComplianceConfig, tokenizer=None, **kwargs):
        super().__init__(config)  # builds self.model = ACT(config) (unmodified), self.reset()
        if config.pretrained_backbone_weights is not None:
            raise ValueError(
                "BiActComplianceConfig.pretrained_backbone_weights must be None -- B4 is the from-scratch "
                "(H4) control; a pretrained ResNet backbone would confound the pretraining comparison."
            )
        self.force_encoder = ForceHistoryEncoder(in_channels=6, hidden_dim=config.force_embed_dim)
        self.lang_encoder = FromScratchLanguageEncoder(
            config.lang_vocab_size, embed_dim=config.lang_embed_dim, out_dim=config.lang_embed_dim
        )
        self.tokenizer = tokenizer  # exposed so scripts/train_policy.py can reuse it for the collate fn, mirrors how b0/b2/b5 expose theirs via policy.model.vlm_with_expert.processor.tokenizer

    def _build_env_state(self, batch):
        force_hist = force_dropout(batch["force_history"], self.config.force_dropout_p, training=self.training)
        force_hist = wrench_bias_injection(force_hist, self.config.wrench_bias_range_n, training=self.training)
        force_emb = self.force_encoder(force_hist)
        lang_emb = self.lang_encoder(batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
        return torch.cat([force_emb, lang_emb], dim=-1)

    def _prep_batch_for_model(self, batch):
        batch = dict(batch)  # shallow copy, mirrors ACTPolicy.forward's own convention -- never mutate the caller's batch
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        batch[OBS_ENV_STATE] = self._build_env_state(batch)
        return batch

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch = self._prep_batch_for_model(batch)
        # ACT's own key/polarity ("action_is_pad", True=pad) differs from this project's dataset convention
        # ("action_valid_mask", True=valid, see src/compliance_vla/policy/dataset.py) -- converted here rather than changing
        # the shared collate function, since B0/B2/B5 already depend on its current key names.
        batch["action_is_pad"] = ~batch["action_valid_mask"]

        actions_hat, (mu, log_sigma_x2) = self.model(batch)

        valid = batch["action_valid_mask"].to(actions_hat.dtype)  # (B, H)
        x_eq_mask = valid[..., None].expand(-1, -1, 6)
        gripper_mask = torch.zeros_like(valid[..., None])  # no gripper channel recorded for T1, see src/compliance_vla/policy/labels.py
        log_k_mask = batch["log_k_mask"].to(actions_hat.dtype) * valid[..., None]  # identifiability AND in-episode-bound

        # See module docstring: compliance_loss's math (residual = arg0 - arg1, masked L1 on x_eq, masked
        # Huber on log_k) is agnostic to flow-matching vs. direct regression -- reused as-is with
        # (target, prediction) in place of (noise, velocity).
        recon_loss, parts = compliance_loss(
            batch[ACTION], actions_hat, x_eq_mask, log_k_mask, gripper_mask,
            lam=self.config.lam_log_k, huber_delta=self.config.huber_delta,
        )
        loss = recon_loss
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
            loss = recon_loss + mean_kld * self.config.kl_weight
            parts["loss_kld"] = mean_kld.item()
        parts["loss"] = loss.item()
        return loss, parts

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        """Inference path (Week 4 evaluation harness) -- same env-state
        construction as forward(), force/language augmentations are no-ops
        outside self.training per force_encoder's own convention."""
        self.eval()
        batch = self._prep_batch_for_model(batch)
        actions = self.model(batch)[0]
        return actions


def load_tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def build_bi_act_policy(config: BiActComplianceConfig, tokenizer=None):
    """scripts/train_policy.py's b4 entry point: resolves the real tokenizer
    vocab size (config.lang_vocab_size starts at the 0 placeholder) before
    constructing the policy, since nn.Embedding needs a fixed vocab size at
    __init__ time."""
    if tokenizer is None:
        tokenizer = load_tokenizer(config.tokenizer_name)
    config.lang_vocab_size = len(tokenizer)
    return BiActCompliancePolicy(config, tokenizer=tokenizer)


def _self_test():
    torch.manual_seed(0)

    # FromScratchLanguageEncoder: shape, gradient flow, padding-invariance, non-degenerate.
    enc = FromScratchLanguageEncoder(vocab_size=50, embed_dim=16, out_dim=16)
    tok = torch.randint(0, 50, (4, 6))
    mask = torch.ones(4, 6, dtype=torch.bool)
    out = enc(tok, mask)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert enc.embed.weight.grad is not None and torch.isfinite(enc.embed.weight.grad).all()

    # padding tokens (masked out) must not change the pooled result.
    tok2 = tok.clone()
    tok2[:, -2:] = 999 % 50  # garbage in the positions about to be masked out
    mask2 = mask.clone()
    mask2[:, -2:] = False
    with torch.no_grad():
        out_a = enc(tok, mask2)
        out_b = enc(tok2, mask2)
    assert torch.allclose(out_a, out_b, atol=1e-5), "masked-out token ids must not affect the pooled embedding"

    # different instructions -> different embeddings (not a degenerate constant encoder).
    tok3 = torch.randint(0, 50, (1, 6))
    tok4 = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        y3, y4 = enc(tok3, mask[:1]), enc(tok4, mask[:1])
    assert not torch.allclose(y3, y4)

    # BiActCompliancePolicy: tiny config for a fast forward/backward smoke check, no real data/GPU needed.
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
        "observation.images.scene_rgb": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        "observation.images.wrist_rgb": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (ENV_STATE_DIM,)),
    }
    output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (13,))}
    norm = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY}
    config = BiActComplianceConfig(
        input_features=input_features, output_features=output_features, normalization_mapping=norm,
        device="cpu", chunk_size=8, n_action_steps=8,
        dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
        use_vae=True, latent_dim=8, n_vae_encoder_layers=1,
    )
    assert config.pretrained_backbone_weights is None, "self-test config must stay from-scratch"

    policy = build_bi_act_policy(config, tokenizer=_FakeTokenizer(vocab_size=37))
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params > 0

    B, H = 3, 8
    batch = {
        "observation.state": torch.randn(B, 20),
        "observation.images.scene_rgb": torch.rand(B, 3, 64, 64),
        "observation.images.wrist_rgb": torch.rand(B, 3, 64, 64),
        "force_history": torch.randn(B, 20, 6),
        "action": torch.randn(B, H, 13),
        "log_k_mask": (torch.rand(B, H, 6) > 0.5),
        "action_valid_mask": torch.ones(B, H, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 37, (B, 5)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(B, 5, dtype=torch.bool),
    }
    loss, parts = policy.forward(batch)
    assert torch.isfinite(loss)
    assert all(v == v and abs(v) != float("inf") for v in parts.values()), f"non-finite loss part: {parts}"
    assert "loss_kld" in parts, "use_vae=True must produce a KL term"

    loss.backward()
    assert policy.force_encoder.net[0].weight.grad is not None, "gradient must reach the force encoder"
    assert torch.isfinite(policy.force_encoder.net[0].weight.grad).all()
    assert policy.lang_encoder.embed.weight.grad is not None, "gradient must reach the language encoder"

    # all-masked-out log_k must not poison the loss (same convention losses.py already tests directly).
    batch_no_mask = dict(batch)
    batch_no_mask["log_k_mask"] = torch.zeros(B, H, 6, dtype=torch.bool)
    loss2, parts2 = policy.forward(batch_no_mask)
    assert parts2["loss_log_k"] == 0.0
    assert torch.isfinite(loss2)

    # inference path: predict_action_chunk returns the right shape, no grad, eval-mode augmentations are no-ops.
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (B, H, 13)
    assert not policy.training, "predict_action_chunk must leave the policy in eval mode"

    print("src/compliance_vla/policy/bi_act_policy.py self-test: PASS")


class _FakeTokenizer:
    """Self-test-only stand-in for transformers.AutoTokenizer -- avoids a
    network/HF-cache dependency in a pure architecture/gradient-flow check.
    build_bi_act_policy only ever calls len() on a real run's tokenizer."""

    def __init__(self, vocab_size):
        self._vocab_size = vocab_size

    def __len__(self):
        return self._vocab_size


if __name__ == "__main__":
    _self_test()
