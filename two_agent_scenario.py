"""
scenario/two_agent.py — v2
==========================
Corrected phase boundary experiment.

The verifier logic in v1 was inverted relative to the theory.

Correct interpretation from the derivation
-------------------------------------------
D⊥* is the MAXIMUM D⊥ that can exist WITHIN a single fiber.
Below D⊥*: a pair of agents can be far apart in D⊥ but still
           in the same fiber — the AR layer cannot distinguish
           them — the composition is UN-verifiable.
Above D⊥*: agents that far apart MUST be in different fibers —
           the AR layer CAN detect the difference — VERIFIABLE.

So the phase transition is:
  D⊥ < D⊥*  →  likely same fiber  →  verifier FAILS (can't distinguish)
  D⊥ > D⊥*  →  likely diff fiber  →  verifier PASSES (can distinguish)

Pass rate goes from 0 (low D⊥) to 1 (high D⊥).
The 50% crossing is D⊥*.

Construction fix
----------------
Sweep agent A's mean from same-fiber territory (below threshold)
to cross-fiber territory (above threshold on some dimensions),
while controlling D⊥ via both mean displacement AND covariance
rotation. This lets D⊥ drive actual fiber crossing.

BLINDNESS preserved: no import of d_perp_star().
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_perp_metric import (
    GaussianAgent,
    d_perp,
    kl_divergence,
    threshold_fiber_id,
    d_perp_decomposed,
)


@dataclass
class ChannelSpec:
    n: int = 4
    sigma: float = 0.3
    thresholds: np.ndarray = None
    B: float = 2.0

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = np.zeros(self.n)
        self.thresholds = np.asarray(self.thresholds, dtype=float)


@dataclass
class TrialResult:
    d_perp_value:    float
    d_kl_value:      float
    perp_factor:     float
    same_fiber:      bool
    invariant_holds: bool
    fiber_A:         tuple
    fiber_B:         tuple


def make_agent_B(n: int, sigma: float) -> GaussianAgent:
    """
    Agent B: fixed reference.
    Mean well above threshold on all dimensions.
    Principal direction: axis 0.
    """
    mu = np.ones(n) * 0.6
    diag = np.ones(n) * (sigma ** 2)
    diag[0] = (sigma * 4.0) ** 2
    return GaussianAgent(mu=mu, Sigma=np.diag(diag), label="B")


def make_agent_A_at_d_perp(
    target_d_perp: float,
    agent_B: GaussianAgent,
    channel: ChannelSpec,
    rng: np.random.Generator,
) -> Tuple[GaussianAgent, float]:
    """
    Constructs agent A such that D⊥(A‖B) ≈ target_d_perp.

    Key insight from the corrected theory:
    - At low D⊥: A is close to B, same fiber — verifier cannot act
    - At high D⊥: A is far from B in perpendicular sense,
      forcing different fibers — verifier can act

    We control D⊥ by jointly varying:
      (a) covariance rotation angle θ (controls perp_factor)
      (b) mean displacement magnitude δ (controls KL component)

    Both together drive D⊥. At large enough D⊥, the mean
    displacement necessarily crosses a threshold boundary,
    putting A in a different fiber from B.
    """
    n = channel.n
    sigma = channel.sigma

    # D⊥ = KL × perp_factor × xi
    # We set perp_factor = sin²(θ) for rotation angle θ ∈ [0, π/2]
    # and KL ≈ 0.5 × (δ/sigma)² for mean displacement δ
    # So: D⊥ ≈ 0.5 × (δ/sigma)² × sin²(θ)
    # We use θ = π/3 (fixed, 60° rotation — substantial but not maximal)
    # and solve for δ.

    theta = np.pi / 3.0  # 60 degrees — fixed orientation
    perp_approx = np.sin(theta) ** 2  # ≈ 0.75

    if target_d_perp < 1e-6:
        delta = 0.0
    else:
        # D⊥ ≈ 0.5 × (delta/sigma)² × perp_approx
        # delta = sigma × sqrt(2 × target_d_perp / perp_approx)
        delta = sigma * np.sqrt(2.0 * target_d_perp / max(perp_approx, 0.01))

    # Place A's mean displaced from B along axis 1
    # (perpendicular to B's principal axis 0)
    # This means the displacement itself is perpendicular to B's
    # principal direction — maximizing D⊥ for given displacement
    mu_A = agent_B.mu.copy()
    mu_A[1] -= delta   # displace in axis-1 direction

    # Add small noise to avoid degenerate configurations
    mu_A += rng.normal(0, sigma * 0.05, n)

    # Covariance: rotate principal direction by theta from axis 0
    base_var = (sigma * 4.0) ** 2
    v = np.zeros(n)
    v[0] = np.cos(theta)
    v[1] = np.sin(theta)
    Sigma_A = (sigma**2) * np.eye(n) + (base_var - sigma**2) * np.outer(v, v)

    agent_A = GaussianAgent(mu=mu_A, Sigma=Sigma_A, label="A")
    result = d_perp(agent_A, agent_B)
    return agent_A, result['d_perp']


class StructuralInvariantVerifier:
    """
    Corrected verifier logic.

    The AR layer checks whether agent A is in a DIFFERENT fiber
    from agent B. If so, the AR layer can formally distinguish
    the two agents' beliefs — the structural invariant (the
    ordering of their beliefs) is VERIFIABLE.

    This is the correct operationalization:
      Same fiber  → invariant cannot be verified (D⊥ within fiber)
      Diff fiber  → invariant CAN be verified (D⊥ between fibers > 0)

    Noise model: the AR layer measures fiber membership with
    Gaussian noise on the threshold comparison (σ_obs = noise_floor).
    """
    def __init__(self, channel: ChannelSpec, noise_floor: float = 0.05):
        self.channel = channel
        self.noise_floor = noise_floor

    def verify(self, agent_A: GaussianAgent, agent_B: GaussianAgent) -> Tuple[bool, dict]:
        decomp = d_perp_decomposed(agent_A, agent_B, self.channel.thresholds)

        # Noisy threshold comparison — the AR layer cannot measure
        # means exactly; add observation noise
        noisy_mu_A = agent_A.mu + np.random.normal(0, self.noise_floor, agent_A.n)
        noisy_fiber_A = threshold_fiber_id(
            GaussianAgent(mu=noisy_mu_A, Sigma=agent_A.Sigma),
            self.channel.thresholds
        )
        fiber_B = threshold_fiber_id(agent_B, self.channel.thresholds)

        different_fiber = (noisy_fiber_A != fiber_B)
        invariant_holds = different_fiber  # verifiable ↔ different fiber

        return invariant_holds, {
            **decomp,
            'fiber_A_noisy': noisy_fiber_A,
            'fiber_B':       fiber_B,
        }


def run_trial(
    target_d_perp: float,
    channel: ChannelSpec,
    verifier: StructuralInvariantVerifier,
    rng: np.random.Generator,
) -> TrialResult:
    agent_B = make_agent_B(channel.n, channel.sigma)
    agent_A, _ = make_agent_A_at_d_perp(target_d_perp, agent_B, channel, rng)

    result_dp = d_perp(agent_A, agent_B)
    invariant_holds, details = verifier.verify(agent_A, agent_B)

    return TrialResult(
        d_perp_value=result_dp['d_perp'],
        d_kl_value=result_dp['d_kl'],
        perp_factor=result_dp['perp_factor'],
        same_fiber=details['same_fiber'],
        invariant_holds=invariant_holds,
        fiber_A=details['fiber_P'],
        fiber_B=details['fiber_Q'],
    )


def _find_crossing(x_vals, y_vals, target=0.5):
    for i in range(len(y_vals) - 1):
        y0, y1 = float(y_vals[i]), float(y_vals[i+1])
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            t = (target - y0) / (y1 - y0 + 1e-12)
            return float(x_vals[i] + t * (x_vals[i+1] - x_vals[i]))
    return float('nan')


def run_phase_diagram(
    d_perp_values: np.ndarray,
    channel: ChannelSpec,
    n_trials: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Blind empirical measurement. No access to d_perp_star()."""
    rng = np.random.default_rng(seed)
    verifier = StructuralInvariantVerifier(channel)

    pass_rates  = []
    kl_means    = []
    dp_achieved = []

    if verbose:
        print(f"\nPhase diagram sweep (v2 — corrected verifier)")
        print(f"  n={channel.n}, sigma={channel.sigma}, B={channel.B}")
        print(f"  {len(d_perp_values)} D⊥ levels × {n_trials} trials each")
        print(f"  {'D⊥_target':>12} {'D⊥_actual':>11} {'KL_mean':>9} "
              f"{'PassRate':>10} {'DiffFiber%':>12}")
        print("  " + "-" * 60)

    for dp_target in d_perp_values:
        trials = [run_trial(dp_target, channel, verifier, rng)
                  for _ in range(n_trials)]

        pr  = np.mean([t.invariant_holds for t in trials])
        klm = np.mean([t.d_kl_value for t in trials])
        dpa = np.mean([t.d_perp_value for t in trials])
        dif = np.mean([not t.same_fiber for t in trials]) * 100

        pass_rates.append(pr)
        kl_means.append(klm)
        dp_achieved.append(dpa)

        if verbose:
            print(f"  {dp_target:>12.4f} {dpa:>11.4f} {klm:>9.4f} "
                  f"{pr:>10.4f} {dif:>11.1f}%")

    pass_rates  = np.array(pass_rates)
    kl_means    = np.array(kl_means)
    dp_achieved = np.array(dp_achieved)

    empirical_d_star = _find_crossing(dp_achieved, pass_rates, 0.5)
    kl_d_star        = _find_crossing(kl_means, pass_rates, 0.5)

    return {
        'd_perp_values':    d_perp_values,
        'dp_achieved':      dp_achieved,
        'pass_rates':       pass_rates,
        'kl_means':         kl_means,
        'empirical_d_star': empirical_d_star,
        'kl_d_star':        kl_d_star,
    }


def main():
    print("=" * 62)
    print("scenario/two_agent.py v2 — Phase Boundary Experiment")
    print("Blind. No access to D⊥* prediction.")
    print("=" * 62)

    channel = ChannelSpec(n=4, sigma=0.3, B=2.0)

    d_perp_grid = np.concatenate([
        np.linspace(0.0, 0.8,  9),
        np.linspace(0.9, 2.2, 27),
        np.linspace(2.3, 3.0,  5),
    ])
    d_perp_grid = np.unique(np.round(d_perp_grid, 4))

    results = run_phase_diagram(
        d_perp_values=d_perp_grid,
        channel=channel,
        n_trials=300,
        seed=42,
        verbose=True,
    )

    print(f"\n{'='*62}")
    print(f"EMPIRICAL RESULT (before comparison to prediction)")
    print(f"{'='*62}")
    print(f"  Empirical D⊥*  = {results['empirical_d_star']:.4f}")
    print(f"  KL crossing    = {results['kl_d_star']:.4f}")
    print(f"\n  Run comparison.py to see gap vs analytical prediction.")
    print(f"{'='*62}")

    return results


if __name__ == "__main__":
    main()
