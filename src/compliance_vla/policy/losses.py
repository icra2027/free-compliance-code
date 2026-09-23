"""Day 11: masked Huber on log k, combined with flow-matching L1 on x_eq.

Proposal §4.2: "flow-matching / L1 on x_eq; masked Huber on log k, weight
lambda tuned on the validation split. Log-space is non-negotiable." and
§4.1: "masked entries excluded, never imputed."

Design choice, not fully specified by the proposal (documented here the same
way M8/evaluate_gate1.py document theirs): SmolVLA's action expert is trained
as conditional flow matching over the *whole* padded action chunk -- there is
no point during training where a direct log_k value is decoded, only the
velocity field v_t predicted from noisy actions x_t at a random corruption
time t (see modeling_smolvla.VLAFlowMatching.forward). The proposal's "masked
Huber on log k" is therefore applied to the same (noise - action) vs
predicted-velocity residual SmolVLA already computes for x_eq, restricted to
the log_k channels and swapping the per-element loss shape from the x_eq
family's L1 to Huber. This keeps a single shared denoising target (required --
the model outputs one joint v_t for the whole 13-dim vector) while giving log
k the robust-to-outlier Huber shape the proposal asks for and x_eq the L1
shape it asks for.
"""

import torch


def masked_huber_loss(residual, mask, delta=1.0):
    """Elementwise Huber loss on `residual`, averaged over `mask`-true entries only.

    residual, mask: same shape, e.g. (B, H, 6). mask is bool or float in {0,1}.
    Masked-out entries are excluded from the mean entirely (never imputed) --
    if `mask` is all-false the returned loss is 0 with a zero (not NaN)
    gradient path, so a batch with zero identifiable log_k timesteps doesn't
    poison the rest of the loss.
    """
    mask = mask.to(dtype=residual.dtype)
    per_elem = torch.nn.functional.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction="none")
    denom = mask.sum()
    if denom.item() == 0:
        return per_elem.sum() * 0.0
    return (per_elem * mask).sum() / denom


def masked_l1_loss(residual, mask):
    """L1 loss on `residual`, averaged over `mask`-true entries only. Same
    masked-mean / zero-safe convention as masked_huber_loss."""
    mask = mask.to(dtype=residual.dtype)
    per_elem = residual.abs()
    denom = mask.sum()
    if denom.item() == 0:
        return per_elem.sum() * 0.0
    return (per_elem * mask).sum() / denom


def position_only_loss(u_t, v_t, x_eq_mask, gripper_mask):
    """B0/B2 baseline loss (Day 12): plain flow-matching L1 on
    [x_eq(6), gripper(1)] -- no log_k term, since these baselines have no
    compliance head (proposal §6.1: B0 = position output, fixed stiffness;
    B2 = force input, position output). Same masked-L1 convention as
    compliance_loss's own x_eq/gripper terms, split out so B0/B2 training
    doesn't need to construct an unused log_k_mask.

    u_t, v_t: (B, H, 7), already sliced to the real dims.
    Returns (total_loss, {"loss_x_eq":..., "loss_gripper":..., "loss":...}).
    """
    residual = u_t - v_t
    loss_x_eq = masked_l1_loss(residual[..., 0:6], x_eq_mask)
    loss_gripper = masked_l1_loss(residual[..., 6:7], gripper_mask)
    total = loss_x_eq + loss_gripper
    return total, {"loss_x_eq": loss_x_eq.item(), "loss_gripper": loss_gripper.item(), "loss": total.item()}


def hybrid_loss(u_t_pos, v_t_pos, x_eq_mask, gripper_mask, log_k_pred, log_k_target, log_k_mask, lam, huber_delta=1.0):
    """Day 14: B3 training loss -- force input + HYBRID force-position output
    (proposal §6.1 B3 row: "ForceVLA2 / Force Policy style", "closest
    competing output parameterization").

    The key difference from compliance_loss (B5): x_eq/gripper are still
    decoded through the flow-matching noise-residual (u_t_pos, v_t_pos,
    exactly like B2's 7-dim position channel), but log_k is decoded by a
    SEPARATE deterministic regression head reading the same suffix
    representation (compliance_vla.policy.hybrid_policy.HybridVLAFlowMatching.log_k_head)
    -- its residual is (prediction - ground-truth value), not
    (noise - action) restricted to the log_k channels. B5 puts x_eq and log_k
    through one shared generative (flow-matching) target; B3 is the
    "hybrid" alternative the proposal names as the closest competing
    parameterization: position via a generative/diffusion-style decoder,
    force/stiffness via a direct regression head off the same backbone --
    matching how ForceVLA2/Force Policy structure their output, per §6.1's
    own one-line description of B3.

    u_t_pos, v_t_pos: (B, H, 7) flow-matching target/prediction, already
        sliced to [x_eq(6), gripper(1)] -- same shape/convention as
        position_only_loss's inputs (B2's loss).
    log_k_pred: (B, H, 6) DIRECT regression output of log_k_head (not a
        noise-residual -- see src/compliance_vla/policy/hybrid_policy.py).
    log_k_target: (B, H, 6) ground-truth log_k (log-space, per §4.1/§4.2 --
        "log-space is non-negotiable").
    log_k_mask: (B, H, 6) -- identifiability AND in-episode-bound, same
        never-imputed convention as compliance_loss.
    lam: loss weight on the log_k term, same role/knob as compliance_loss's
        lam (tuned on the validation split, §4.2).

    Returns (total_loss, {"loss_x_eq":..., "loss_log_k":..., "loss_gripper":...}).
    """
    residual_pos = u_t_pos - v_t_pos
    loss_x_eq = masked_l1_loss(residual_pos[..., 0:6], x_eq_mask)
    loss_gripper = masked_l1_loss(residual_pos[..., 6:7], gripper_mask)
    log_k_residual = log_k_pred - log_k_target  # direct regression residual, NOT a flow-matching noise residual
    loss_log_k = masked_huber_loss(log_k_residual, log_k_mask, delta=huber_delta)
    total = loss_x_eq + lam * loss_log_k + loss_gripper
    return total, {
        "loss_x_eq": loss_x_eq.item(),
        "loss_log_k": loss_log_k.item(),
        "loss_gripper": loss_gripper.item(),
        "loss": total.item(),
    }


def compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam, huber_delta=1.0):
    """Combined B5 training loss.

    u_t, v_t: (B, H, 13) target/predicted flow-matching velocity, already
        sliced down to the real 13 dims (x_eq[0:6], log_k[6:12], gripper[12]).
    x_eq_mask, gripper_mask: (B, H, *) float/bool, in-episode-bound mask
        (excludes steps past the end of a short episode). gripper_mask is
        all-zero for this T1-only dataset (see src/compliance_vla/policy/labels.py) -- the
        gripper head exists for architectural parity with the 13-dim spec and
        future tasks, it is not supervised here.
    log_k_mask: (B, H, 6) -- x_eq_mask/in-episode-bound AND the extraction
        pipeline's per-axis identifiability mask (never imputed).
    lam: loss weight on the log_k term, tuned on the validation split per
        §4.2 (left as a CLI/config knob, see scripts/train_b5.py --lam).

    Returns (total_loss, {"loss_x_eq":..., "loss_log_k":..., "loss_gripper":...}).
    """
    residual = u_t - v_t
    loss_x_eq = masked_l1_loss(residual[..., 0:6], x_eq_mask)
    loss_log_k = masked_huber_loss(residual[..., 6:12], log_k_mask, delta=huber_delta)
    loss_gripper = masked_l1_loss(residual[..., 12:13], gripper_mask)
    total = loss_x_eq + lam * loss_log_k + loss_gripper
    return total, {
        "loss_x_eq": loss_x_eq.item(),
        "loss_log_k": loss_log_k.item(),
        "loss_gripper": loss_gripper.item(),
        "loss": total.item(),
    }


def _self_test():
    torch.manual_seed(0)

    # masked_huber_loss: closed-form check against a hand-built example.
    residual = torch.tensor([[1.0, 5.0, 0.1], [0.5, 0.5, 0.5]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    delta = 1.0
    # Huber(1.0)=0.5, Huber(5.0) excluded, Huber(0.1)=0.005, Huber(0.5)=0.125 (x3, one masked out... none masked here)
    expected_terms = [0.5, 0.005, 0.125, 0.125, 0.125]  # (0,0),(0,2),(1,0),(1,1),(1,2); (0,1) excluded
    expected = sum(expected_terms) / len(expected_terms)
    got = masked_huber_loss(residual, mask, delta=delta).item()
    assert abs(got - expected) < 1e-6, f"masked_huber_loss mismatch: {got} vs {expected}"

    # all-masked-out -> exactly zero, not NaN.
    zero_mask = torch.zeros(3, 4)
    got_zero = masked_huber_loss(torch.randn(3, 4), zero_mask)
    assert got_zero.item() == 0.0, "all-false mask should give exactly 0.0"
    assert not torch.isnan(got_zero), "all-false mask produced NaN"

    # masked_l1_loss closed-form.
    r = torch.tensor([[2.0, -3.0], [1.0, 1.0]])
    m = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    expected_l1 = (2.0 + 3.0 + 1.0) / 3.0
    got_l1 = masked_l1_loss(r, m).item()
    assert abs(got_l1 - expected_l1) < 1e-6, f"masked_l1_loss mismatch: {got_l1} vs {expected_l1}"

    # compliance_loss: gripper term with all-zero mask must not contribute (and must not NaN).
    B, H = 4, 8
    u_t = torch.randn(B, H, 13)
    v_t = torch.randn(B, H, 13)
    x_eq_mask = torch.ones(B, H, 6)
    log_k_mask = (torch.rand(B, H, 6) > 0.7).float()  # sparse, like real identifiability coverage
    gripper_mask = torch.zeros(B, H, 1)
    total, parts = compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam=0.5)
    assert parts["loss_gripper"] == 0.0
    assert abs(parts["loss"] - (parts["loss_x_eq"] + 0.5 * parts["loss_log_k"] + parts["loss_gripper"])) < 1e-5
    assert not any(v != v for v in parts.values()), f"NaN in loss parts: {parts}"

    # lam actually changes the total.
    total_lo, _ = compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam=0.0)
    total_hi, _ = compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam=10.0)
    assert total_hi.item() > total_lo.item(), "increasing lam should increase total loss when log_k residual != 0"

    # position_only_loss: matches compliance_loss's x_eq+gripper terms exactly (B0/B2 baselines).
    u7, v7 = torch.randn(B, H, 7), torch.randn(B, H, 7)
    gripper_mask7 = torch.zeros(B, H, 1)
    total_pos, parts_pos = position_only_loss(u7, v7, x_eq_mask, gripper_mask7)
    assert parts_pos["loss_gripper"] == 0.0
    assert abs(parts_pos["loss"] - parts_pos["loss_x_eq"]) < 1e-6, "gripper term is masked out, should not shift total"
    assert not any(v != v for v in parts_pos.values())

    # hybrid_loss (B3, Day 14): position term matches position_only_loss exactly (same
    # u7/v7/x_eq_mask/gripper_mask7 inputs) -- the only thing that should differ from B2's
    # loss is the added, separately-supervised log_k regression term.
    log_k_pred = torch.randn(B, H, 6)
    log_k_target = torch.randn(B, H, 6)
    total_hyb, parts_hyb = hybrid_loss(
        u7, v7, x_eq_mask, gripper_mask7, log_k_pred, log_k_target, log_k_mask, lam=0.5,
    )
    assert abs(parts_hyb["loss_x_eq"] - parts_pos["loss_x_eq"]) < 1e-6, \
        "hybrid_loss's position term should equal position_only_loss's given the same u7/v7 inputs"
    assert parts_hyb["loss_gripper"] == 0.0
    assert not any(v != v for v in parts_hyb.values()), f"NaN in hybrid_loss parts: {parts_hyb}"

    # hybrid_loss's log_k term is a DIRECT regression residual (pred - target), not a
    # flow-matching noise residual -- confirm it actually differs from what compliance_loss
    # would compute on the same log_k_mask if log_k_pred/target were (incorrectly) treated as
    # a u_t/v_t pair, i.e. confirm this isn't an accidental reimplementation of B5's loss.
    _, parts_as_noise_residual = compliance_loss(
        torch.cat([u7[..., 0:6], log_k_target, u7[..., 6:7]], dim=-1),
        torch.cat([v7[..., 0:6], log_k_pred, v7[..., 6:7]], dim=-1),
        x_eq_mask, log_k_mask, gripper_mask, lam=0.5,
    )
    # loss_log_k in both cases reduces to masked_huber_loss(log_k_pred - log_k_target, log_k_mask)
    # -- same formula, so the numeric value legitimately matches; what differs (documented in
    # hybrid_policy.py) is what produces log_k_pred at inference time (a one-shot regression
    # head vs. num_steps of iterative flow-matching denoising), which this loss-level test
    # cannot see. Assert the formula match instead, as the thing this test *can* check.
    assert abs(parts_hyb["loss_log_k"] - parts_as_noise_residual["loss_log_k"]) < 1e-6

    # lam actually changes hybrid_loss's total, same contract as compliance_loss's lam check.
    total_hyb_lo, _ = hybrid_loss(u7, v7, x_eq_mask, gripper_mask7, log_k_pred, log_k_target, log_k_mask, lam=0.0)
    total_hyb_hi, _ = hybrid_loss(u7, v7, x_eq_mask, gripper_mask7, log_k_pred, log_k_target, log_k_mask, lam=10.0)
    assert total_hyb_hi.item() > total_hyb_lo.item(), "increasing lam should increase hybrid_loss's total"

    print("src/compliance_vla/policy/losses.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
