"""Day 11/12: B2 and B5 policies -- shared force-input machinery, split by
`use_compliance_head` (proposal §6.1 baseline table).

  - B0 (position output, fixed stiffness, no force input) needs none of this
    module's additions -- it is lerobot's unmodified `SmolVLAConfig`/
    `SmolVLAPolicy`, used directly, see scripts/train_policy.py. Included
    here only as a comment so the three baselines' relationship is visible
    from one place.
  - B2 (force input, position output): `ComplianceSmolVLAConfig` with
    `use_force_input=True, use_compliance_head=False` -- registered as
    `B2Config` below. 7-dim action ([x_eq(6), gripper(1)]), plain
    flow-matching loss (compliance_vla.policy.losses.position_only_loss), force token
    injected exactly like B5's.
  - B5 (force input, compliance output, "ours"): `ComplianceSmolVLAConfig`
    defaults (`use_force_input=True, use_compliance_head=True`). 13-dim
    compliance output head + force-history token injected post-VLM
    (proposal §4.2), as built on Day 11.

Architecture summary (see the ASCII diagram in modeling_smolvla.VLAFlowMatching
for the unmodified base):
  - VLM prefix (images + language + state) is untouched -- same
    `embed_prefix`, same frozen/finetuned SmolVLM2 backbone as base SmolVLA.
  - The force-history token is a *new first token in the action-expert's own
    "suffix" sequence*, added by ForceHistoryEncoder (1D-conv over the last
    500ms of 6-DoF wrench, see force_encoder.py) and placed before the
    action-time tokens in embed_suffix -- i.e. after the VLM has already run
    (`embed_prefix`/the VLM forward pass never sees it), matching the
    proposal's "injected post-VLM" wording and the ForceVLA finding it
    cites. It gets the same attention-mask "block start" bit the state token
    already gets, so it is visible to the VLM prefix + itself only as a
    query, but every action-chunk token (later in the same suffix, larger
    cumulative attention-mask index) can see it as a key -- global context
    for the whole H=32 chunk, not injected per-timestep.
  - Output stays 32 action-expert tokens -> action_out_proj -> a
    (B, chunk_size, max_action_dim) velocity field, same as base SmolVLA;
    only the first 13 dims ([x_eq(6), log_k(6), gripper(1)]) are real, the
    rest is SmolVLA's own max_action_dim=32 padding.

`ComplianceVLAFlowMatching.forward/sample_actions/denoise_step` are
deliberate, minimal copies of the upstream methods (not overridable via a
single hook -- the upstream `embed_suffix` call sites don't accept an extra
argument) with one line changed each: `embed_suffix(...)` ->
`embed_suffix(..., force_emb=force_emb)`. Diff against modeling_smolvla.py
if upstream changes.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .force_encoder import ForceHistoryEncoder, force_dropout, wrench_bias_injection
from .losses import compliance_loss, position_only_loss


@PreTrainedConfig.register_subclass("smolvla_compliance")
@dataclass
class ComplianceSmolVLAConfig(SmolVLAConfig):
    # H = 32 @ 30Hz (~1.07s), proposal §4.2 -- overrides SmolVLAConfig's defaults (50/50).
    chunk_size: int = 32
    n_action_steps: int = 32

    # Wrist/scene RGB are captured at 224x224 (proposal §4.2), but
    # SmolVLM2's own SigLIP vision encoder expects a 512x512-shaped patch
    # grid internally (its pixel_shuffle connector reshapes on that
    # assumption -- a real RuntimeError, not a preference, when fed a
    # 224x224-shaped patch grid instead). Deliberately *not* overriding
    # SmolVLAConfig's tested resize_imgs_with_padding=(512, 512) default:
    # `resize_with_pad` upsamples our 224x224 frames to match, same as
    # base SmolVLA does for any camera that isn't natively 512x512.

    # Force-history token (§4.2) and its training-time augmentations (§4.2).
    force_history_len: int = 20
    force_history_window_sec: float = 0.5
    force_dropout_p: float = 0.15
    wrench_bias_range_n: float = 1.0

    # Compliance-head loss (§4.2): masked Huber on log_k, weight tuned on val.
    huber_delta: float = 1.0
    lam_log_k: float = 1.0

    # Day 12: which baseline this config builds. B5 = both True (default,
    # unchanged from Day 11). B2 sets use_compliance_head=False via the
    # B2Config subclass below. B0 doesn't use this config class at all.
    use_force_input: bool = True
    use_compliance_head: bool = True


@PreTrainedConfig.register_subclass("smolvla_b2_force_position")
@dataclass
class B2Config(ComplianceSmolVLAConfig):
    """B2: VLA + force as input, position output (proposal §6.1) -- same
    force-injection machinery as B5, 7-dim position-only output instead of
    the 13-dim compliance head."""

    use_force_input: bool = True
    use_compliance_head: bool = False


class ComplianceVLAFlowMatching(VLAFlowMatching):
    def __init__(self, config: ComplianceSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.force_encoder = (
            ForceHistoryEncoder(in_channels=6, hidden_dim=self.vlm_with_expert.expert_hidden_size)
            if config.use_force_input else None
        )

    def embed_suffix(self, noisy_actions, timestep, force_emb=None):
        """Same as VLAFlowMatching.embed_suffix, with one addition: a force
        token prepended before the action-time tokens when force_emb is
        given. `forward`/`denoise_step` both slice `suffix_out[:,
        -chunk_size:]` afterwards, which is why the force token must go
        *first*, not appended at the end."""
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        if force_emb is not None:
            force_tok = force_emb[:, None, :].to(dtype=dtype)
            embs.append(force_tok)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.vlm_with_expert.expert_hidden_size, self.config.min_period,
            self.config.max_period, device=device,
        ).type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)

        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device))
        att_masks += [1] * self.config.chunk_size

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(  # noqa: D102 -- deliberate copy of VLAFlowMatching.forward, see module docstring
        self, images, img_masks, lang_tokens, lang_masks, state, actions, force_hist=None, noise=None, time=None
    ) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        force_emb = self.force_encoder(force_hist) if self.force_encoder is not None else None

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time, force_emb=force_emb)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return u_t, v_t

    def sample_actions(  # noqa: D102 -- deliberate copy of VLAFlowMatching.sample_actions (RTC path dropped, out of Day-11 scope)
        self, images, img_masks, lang_tokens, lang_masks, state, force_hist=None, noise=None, **kwargs
    ) -> Tensor:
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        force_emb = self.force_encoder(force_hist) if self.force_encoder is not None else None

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=self.config.use_cache, fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                x_t=x_t, timestep=time_tensor, force_emb=force_emb,
            )
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, force_emb):  # noqa: D102
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep, force_emb=force_emb)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks, position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs], use_cache=self.config.use_cache, fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class ComplianceSmolVLAPolicy(SmolVLAPolicy):
    """B2 (force input, position output) or B5 (force input, compliance
    output), selected by config.use_compliance_head -- see module docstring.
    """

    config_class = ComplianceSmolVLAConfig
    name = "smolvla_compliance"

    def __init__(self, config: ComplianceSmolVLAConfig, **kwargs):
        # Deliberately does not call SmolVLAPolicy.__init__: it hardcodes
        # VLAFlowMatching. Replicates its body with ComplianceVLAFlowMatching instead.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ComplianceVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def prepare_force_history(self, batch):
        return batch["force_history"] if self.config.use_force_input else None

    def _real_action_dim(self):
        return 13 if self.config.use_compliance_head else 7

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        force_hist = self.prepare_force_history(batch)
        if force_hist is not None:
            force_hist = force_dropout(force_hist, self.config.force_dropout_p, training=self.training)
            force_hist = wrench_bias_injection(force_hist, self.config.wrench_bias_range_n, training=self.training)

        u_t, v_t = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, force_hist, noise, time
        )
        real_dim = self._real_action_dim()
        u_t = u_t[:, :, :real_dim]
        v_t = v_t[:, :, :real_dim]

        valid = batch["action_valid_mask"].to(u_t.dtype)  # (B, H) -- False past episode end
        x_eq_mask = valid[..., None].expand(-1, -1, 6)
        gripper_mask = torch.zeros_like(valid[..., None])  # no gripper channel recorded for T1, see labels.py

        if self.config.use_compliance_head:
            log_k_mask = batch["log_k_mask"].to(u_t.dtype) * valid[..., None]  # identifiability AND in-episode-bound
            loss, parts = compliance_loss(
                u_t, v_t, x_eq_mask, log_k_mask, gripper_mask,
                lam=self.config.lam_log_k, huber_delta=self.config.huber_delta,
            )
        else:
            loss, parts = position_only_loss(u_t, v_t, x_eq_mask, gripper_mask)
        return loss, parts

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        force_hist = self.prepare_force_history(batch)  # eval/inference: augmentations are no-ops (self.training=False)
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, force_hist, noise=noise, **kwargs
        )
        return actions[:, :, :self._real_action_dim()]


