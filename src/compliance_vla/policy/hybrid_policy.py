"""Day 14: B3 -- force input + HYBRID force-position output (proposal §6.1:
"ForceVLA2 / Force Policy style", "closest competing output parameterization").

Where this sits relative to the other force-input baselines already built
(src/compliance_vla/policy/compliance_policy.py, Day 11/12):
  - B2 (force in, position out): flow-matching on a 7-dim [x_eq(6),
    gripper(1)] action, no compliance output at all.
  - B5 ("ours", force in, compliance out): flow-matching on the FULL 13-dim
    [x_eq(6), log_k(6), gripper(1)] action -- x_eq and log_k are decoded
    through the SAME shared noise-residual target, split only at the loss
    (masked L1 for x_eq, masked Huber for log_k), see compliance_policy.py's
    module docstring.
  - B3 (this module): force in, HYBRID force-position out. Position/gripper
    are decoded exactly like B2 -- iterative flow-matching over a 7-dim
    action. log_k is decoded by a SEPARATE, one-shot deterministic
    regression head (`log_k_head`, a plain nn.Linear) reading the same
    suffix transformer representation used to produce the terminal position
    action, supervised with a direct (prediction - target) Huber residual,
    never routed through the noise process. This is what ForceVLA2 (arXiv
    2603.15169) and Force Policy (RSS 2026, 2602.22088) actually do:
    generative/diffusion decoding for the continuous pose trajectory, a
    separate direct-regression (or control-parameter) head for force/
    stiffness -- structurally distinct from B5's single joint generative
    target for both channels, which is exactly the ablation the proposal's
    baseline table (§6.1) asks B3 to isolate ("closest competing output
    parameterization").

Architecturally otherwise identical to B2/B5 (§4.2's "keep the architecture
boring on purpose" applies here too): same force-history token, injected
post-VLM the same way (reuses ComplianceVLAFlowMatching.embed_suffix
unmodified), same augmentations (force dropout, wrench-bias injection), same
pretrained SmolVLM2 backbone. The only change is where log_k comes from.

`HybridVLAFlowMatching.forward/sample_actions/denoise_step` are, like
ComplianceVLAFlowMatching's, deliberate minimal copies of the upstream
methods -- kept as copies (not refactored into one parameterized function)
because compliance_policy.py's own docstring already establishes that
precedent for this exact reason (upstream's `embed_suffix` call sites don't
accept extra arguments) and this module should diff cleanly against that one
rather than introduce a second abstraction style.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .compliance_policy import ComplianceSmolVLAConfig, ComplianceSmolVLAPolicy, ComplianceVLAFlowMatching
from .force_encoder import force_dropout, wrench_bias_injection
from .losses import hybrid_loss, position_only_loss

POS_ACTION_DIM = 7  # [x_eq(6), gripper(1)] -- the only channels that go through flow matching in B3


@PreTrainedConfig.register_subclass("smolvla_b3_hybrid")
@dataclass
class HybridSmolVLAConfig(ComplianceSmolVLAConfig):
    """B3: force input, hybrid force-position output (proposal §6.1).

    Inherits every B2/B5-shared field (chunk_size, force_history_*,
    force_dropout_p, wrench_bias_range_n, huber_delta, lam_log_k) from
    ComplianceSmolVLAConfig unchanged -- only `use_compliance_head` differs
    in meaning here: for HybridSmolVLAPolicy it selects "route log_k through
    the direct regression head" rather than "extend the flow-matching action
    vector to 13 dims" (B5's meaning). Kept True as the default so B3, like
    B5, always predicts SOME log_k output (the point of the baseline is to
    compare output *parameterizations* for compliance, not to reproduce B2 a
    second time) -- flip to False only for debugging.
    """

    use_force_input: bool = True
    use_compliance_head: bool = True


class HybridVLAFlowMatching(ComplianceVLAFlowMatching):
    def __init__(self, config: HybridSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.log_k_head = nn.Linear(self.vlm_with_expert.expert_hidden_size, 6)

    def forward(  # noqa: D102 -- deliberate copy of ComplianceVLAFlowMatching.forward, see module docstring
        self, images, img_masks, lang_tokens, lang_masks, state, pos_actions, force_hist=None, noise=None, time=None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Same as ComplianceVLAFlowMatching.forward, except:
          - `pos_actions` is already the 7-dim [x_eq(6), gripper(1)] slice
            (padded to max_action_dim by the caller, exactly like B2's
            `actions` -- see HybridSmolVLAPolicy.forward) -- flow matching
            never sees log_k at all, so there is no noise/denoising target
            for it to corrupt.
          - a third return value, `log_k_pred` (B, H, 6): the direct
            regression head's output, computed from the SAME suffix
            transformer features (`suffix_out`, pre-action_out_proj) used to
            decode `v_t` -- one shared backbone pass, two decoder heads.
        """
        if noise is None:
            noise = self.sample_noise(pos_actions.shape, pos_actions.device)
        if time is None:
            time = self.sample_time(pos_actions.shape[0], pos_actions.device)

        force_emb = self.force_encoder(force_hist) if self.force_encoder is not None else None

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * pos_actions
        u_t = noise - pos_actions
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
        log_k_pred = self.log_k_head(suffix_out)
        return u_t, v_t, log_k_pred

    def sample_actions(  # noqa: D102 -- deliberate copy of ComplianceVLAFlowMatching.sample_actions
        self, images, img_masks, lang_tokens, lang_masks, state, force_hist=None, noise=None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Returns (pos_action_chunk, log_k_pred). Position undergoes the
        normal iterative flow-matching denoising loop (7-dim, padded to
        max_action_dim internally by embed_suffix's action_in_proj). log_k
        needs no iteration (it is a direct regression, not a generative
        target) -- `denoise_step` computes it on every call anyway (cheap,
        one nn.Linear) since that avoids threading a "last step only" flag
        through the loop; this method keeps only the FINAL call's log_k_pred,
        i.e. the regression head reads the suffix representation conditioned
        on the fully-denoised (t=0) position action -- the same terminal
        representation that produced the position output it's paired with.
        """
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
        log_k_pred = None
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t, log_k_pred = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                x_t=x_t, timestep=time_tensor, force_emb=force_emb,
            )
            x_t = x_t + dt * v_t
        return x_t, log_k_pred

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
        return self.action_out_proj(suffix_out), self.log_k_head(suffix_out)


class HybridSmolVLAPolicy(ComplianceSmolVLAPolicy):
    """B3 (force input, hybrid force-position output) -- see module docstring."""

    config_class = HybridSmolVLAConfig
    name = "smolvla_b3_hybrid"

    def __init__(self, config: HybridSmolVLAConfig, **kwargs):
        # Deliberately does not call ComplianceSmolVLAPolicy.__init__ (it hardcodes
        # ComplianceVLAFlowMatching) -- replicates its body with HybridVLAFlowMatching instead.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = HybridVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)  # (B, H, max_action_dim), real content in [0:13]

        # Flow matching only ever sees the 7-dim position/gripper slice -- log_k (cols 6:12)
        # is read straight off the ground-truth action tensor as a direct regression target,
        # never corrupted with noise (that's the whole point of "hybrid": position is
        # generative, log_k is not). pos_actions is re-padded to max_action_dim so it fits
        # action_in_proj's fixed input width, same as B2's already-padded 7-dim actions do.
        real_dim = self._real_action_dim()  # 13 for this config (use_compliance_head=True)
        log_k_target = actions[..., 6:12]
        pos_actions = torch.cat([actions[..., 0:6], actions[..., 12:13]], dim=-1)
        pos_actions = torch.nn.functional.pad(pos_actions, (0, actions.shape[-1] - POS_ACTION_DIM))

        force_hist = self.prepare_force_history(batch)
        if force_hist is not None:
            force_hist = force_dropout(force_hist, self.config.force_dropout_p, training=self.training)
            force_hist = wrench_bias_injection(force_hist, self.config.wrench_bias_range_n, training=self.training)

        u_t, v_t, log_k_pred = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, pos_actions, force_hist, noise, time
        )
        u_t_pos = u_t[:, :, :POS_ACTION_DIM]
        v_t_pos = v_t[:, :, :POS_ACTION_DIM]

        valid = batch["action_valid_mask"].to(u_t.dtype)  # (B, H) -- False past episode end
        x_eq_mask = valid[..., None].expand(-1, -1, 6)
        gripper_mask = torch.zeros_like(valid[..., None])  # no gripper channel recorded for T1, see labels.py

        if real_dim == POS_ACTION_DIM:
            # use_compliance_head=False debug path: identical to B2's loss, log_k unused.
            return position_only_loss(u_t_pos, v_t_pos, x_eq_mask, gripper_mask)

        log_k_mask = batch["log_k_mask"].to(u_t.dtype) * valid[..., None]  # identifiability AND in-episode-bound
        loss, parts = hybrid_loss(
            u_t_pos, v_t_pos, x_eq_mask, gripper_mask, log_k_pred, log_k_target, log_k_mask,
            lam=self.config.lam_log_k, huber_delta=self.config.huber_delta,
        )
        return loss, parts

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        force_hist = self.prepare_force_history(batch)  # eval/inference: augmentations are no-ops (self.training=False)
        pos_action_chunk, log_k_pred = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, force_hist, noise=noise, **kwargs
        )
        # Reassemble the 13-dim [x_eq(6), log_k(6), gripper(1)] layout the rest of this
        # project's tooling (serve_policy.py, the low-level controller) expects, from the
        # two decoder heads' separate outputs.
        x_eq = pos_action_chunk[:, :, 0:6]
        gripper = pos_action_chunk[:, :, 6:7]
        return torch.cat([x_eq, log_k_pred[:, :, :6], gripper], dim=-1)


def _self_test():
    torch.manual_seed(0)
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (20,)),
        "observation.images.scene_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        "observation.images.wrist_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    }
    output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (13,))}
    normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY,
    }
    config = HybridSmolVLAConfig(
        input_features=input_features, output_features=output_features,
        normalization_mapping=normalization_mapping, device="cpu",
        load_vlm_weights=False,  # self-test: architecture/shape/gradient-flow check only, no HF download
        chunk_size=4, n_action_steps=4, num_steps=2,
    )
    policy = HybridSmolVLAPolicy(config)
    policy.train()

    B, H = 2, 4
    batch = {
        "observation.state": torch.randn(B, 20),
        "observation.images.scene_rgb": torch.rand(B, 3, 224, 224),
        "observation.images.wrist_rgb": torch.rand(B, 3, 224, 224),
        "observation.language.tokens": torch.randint(0, 100, (B, 8)),
        "observation.language.attention_mask": torch.ones(B, 8, dtype=torch.bool),
        "action": torch.randn(B, H, 13),
        "log_k_mask": (torch.rand(B, H, 6) > 0.4),
        "action_valid_mask": torch.ones(B, H, dtype=torch.bool),
        "force_history": torch.randn(B, 20, 6),
    }

    loss, parts = policy.forward(batch)
    assert torch.isfinite(loss), f"non-finite loss: {parts}"
    assert not any(v != v for v in parts.values()), f"NaN in loss parts: {parts}"
    assert set(parts.keys()) == {"loss_x_eq", "loss_log_k", "loss_gripper", "loss"}, parts

    # gradients reach both decoder heads (the whole point of "hybrid": one backbone, two heads)
    # and the force encoder (force input must be load-bearing here too, same as B2/B5).
    loss.backward()
    assert policy.model.log_k_head.weight.grad is not None, "log_k_head got no gradient"
    assert torch.isfinite(policy.model.log_k_head.weight.grad).all()
    assert policy.model.action_out_proj.weight.grad is not None, "action_out_proj (position head) got no gradient"
    assert policy.model.force_encoder.out_proj.weight.grad is not None, "force encoder got no gradient"

    # all-masked-out log_k -> exactly zero log_k loss, not NaN (never-imputed convention).
    batch_no_mask = dict(batch, log_k_mask=torch.zeros(B, H, 6, dtype=torch.bool))
    policy.zero_grad()
    loss_nomask, parts_nomask = policy.forward(batch_no_mask)
    assert parts_nomask["loss_log_k"] == 0.0
    assert torch.isfinite(loss_nomask)

    # inference path: shape and finiteness, and log_k is NOT routed through the flow-matching
    # noise process at inference either (checked structurally: sample_actions returns it
    # straight from the regression head, not from x_t's iterative update).
    policy.eval()
    with torch.no_grad():
        action_chunk = policy._get_action_chunk(batch)
    assert action_chunk.shape == (B, H, 13), action_chunk.shape
    assert torch.isfinite(action_chunk).all()

    print("src/compliance_vla/policy/hybrid_policy.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
