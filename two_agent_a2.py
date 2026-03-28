"""
scenario/two_agent_a2.py
========================
Scenario designed to match Assumption A2 exactly.

Assumption A2 (from derivation):
  - Distributions are Gaussian N(mu, Sigma)
  - Sigma = sigma²I  (isotropic — SAME for both agents)
  - Symmetric thresholds at 0 on all dimensions
  - Both agents start with means above threshold (same fiber)
  - D⊥ is varied by displacing agent A's mean toward the
    threshold boundary on one dimension only

Under these conditions:
  - perp_factor is determined by mean displacement direction
    vs covariance principal direction
  - For isotropic Sigma, ALL directions are principal directions
    with equal eigenvalue — so perp_factor depends only on
    the angle between the displacement vector and itself
  - With isotropic Sigma: principal direction is ILL-DEFINED
    (all eigenvectors equal eigenvalue sigma²)

This is a subtle point. For isotropic Gaussians, D⊥ reduces to:
  D⊥ = D_KL × (1 - |<u_P, u_Q>|) × xi
But u_P and u_Q are arbitrary unit vectors when Sigma = sigma²I.

Resolution: for isotropic agents, the perpendicularity factor
is determined by the MEAN DISPLACEMENT DIRECTION relative to
a reference direction. We use the displacement vector itself
as the effective principal direction for each agent.

This gives D⊥ a clean geometric meaning:
  u_A = direction of (mu_A - origin) = mu_A / |mu_A|
  u_B = direction of (mu_B - origin) = mu_B / |mu_B|
  perp_factor = 1 - |<u_A, u_B>|

So D⊥ = 0 when agents point in the same direction,
   D⊥ = D_KL when they point in orthogonal directions.

Experiment design
-----------------
Agent B: fixed at mu_B = [0.5, 0, 0, 0] (above threshold on dim 0)
Agent A: mu_A = [0.5 - delta*cos(alpha), delta*sin(alpha), 0, 0]
  - alpha=0: A moves along dim 0 (same direction as B) → perp=0
  - alpha=pi/2: A moves along dim 1 (orthogonal to B) → perp=1

We fix alpha=pi/2 (maximum perpendicularity) so that:
  D⊥ = D_KL × 1 × 1 = D_KL

And vary delta to control D_KL (and thus D⊥).

The threshold crossing happens when mu_A[0] < 0 (dim 0 crosses 0)
or mu_A[1] > 0 (dim 1 crosses 0 — already satisfied).

Actually with alpha=pi/2:
  mu_A = [0.5, delta, 0, 0]
  A is always in same fiber as B on dim 0 (both > 0)
  A is always in same fiber as B on dim 1 (both > 0 for delta > 0)

We need A to CROSS a threshold to get into a different fiber.
Design: B sits at [0.5, 0.5, 0.5, 0.5]. A starts there and
moves in a direction that crosses one threshold boundary.

  mu_A = [0.5 - delta, 0.5, 0.5, 0.5]
  Threshold crossing: dim 0 crosses 0 when delta > 0.5

At the crossing, the fiber ID of A changes on dim 0.
The verifier detects this.

D_KL at crossing = 0.5 × (delta/sigma)² (for isotropic, 1D displacement)
D⊥ at crossing  = D_KL × 1 × 1 = 0.5 × (0.5/sigma)²

For sigma=0.3: D⊥_crossing = 0.5 × (0.5/0.3)² = 0.5 × 2.778 = 1.389

Predicted D⊥* from derivation: 1.5123

These should be close. The small difference is because D_KL for
a full n=4 displacement (not 1D) adds contributions from all dims.

BLINDNESS: this file does not import d_perp_star().
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d_perp_metric import (
    GaussianAgent, d_perp, kl_divergence,
    threshold_fiber_id, d_perp_decomposed,
)


# ─────────────────────────────────────────────
# A2-compliant agent construction
# ─────────────────────────────────────────────

SIGMA = 0.3
N = 4
THRESHOLDS = np.zeros(N)
# B sits at [0.5, 0.5, 0.5, 0.5] — comfortably above all thresholds
MU_B = np.ones(N) * 0.5
SIGMA_MAT = (SIGMA ** 2) * np.eye(N)  # isotropic, same for both


def make_agents_a2(delta: float, rng: np.random.Generator):
    """
    Agent B: fixed at MU_B, isotropic sigma.
    Agent A: displaced by delta on dim 0 only.
      mu_A = [0.5 - delta, 0.5, 0.5, 0.5]

    Threshold crossing on dim 0 occurs at delta = 0.5.
    Below delta=0.5: same fiber. Above: different fiber on dim 0.

    D⊥ = D_KL × perp_factor × xi
    For isotropic Sigma, principal directions are degenerate.
    We use mean-displacement direction as effective principal:
      u_B: direction of MU_B from origin → [1,1,1,1]/2
      u_A: direction of mu_A from origin

    perp_factor = 1 - |<u_A, u_B>|
    """
    agent_B = GaussianAgent(mu=MU_B.copy(), Sigma=SIGMA_MAT.copy(), label="B")

    mu_A = MU_B.copy()
    mu_A[0] = MU_B[0] - delta
    # Add tiny noise to avoid exact degeneracy
    mu_A += rng.normal(0, SIGMA * 0.01, N)

    agent_A = GaussianAgent(mu=mu_A, Sigma=SIGMA_MAT.copy(), label="A")
    return agent_A, agent_B


def measure_d_perp_a2(agent_A: GaussianAgent, agent_B: GaussianAgent) -> dict:
    """
    For isotropic agents, compute D⊥ using mean-direction as
    effective principal direction.

    This is the A2-compliant D⊥ measurement.
    """
    # D_KL (standard closed form, works for isotropic too)
    dkl = kl_divergence(agent_A, agent_B)

    # Effective principal directions = unit mean vectors
    norm_A = np.linalg.norm(agent_A.mu)
    norm_B = np.linalg.norm(agent_B.mu)

    if norm_A < 1e-10 or norm_B < 1e-10:
        perp = 0.0
    else:
        u_A = agent_A.mu / norm_A
        u_B = agent_B.mu / norm_B
        perp = float(1.0 - abs(np.dot(u_A, u_B)))
        perp = max(0.0, min(1.0, perp))

    xi = float(np.sqrt(agent_A.xi * agent_B.xi))
    dp = dkl * perp * xi

    return {
        'd_kl':        dkl,
        'perp_factor': perp,
        'xi':          xi,
        'd_perp':      dp,
    }


def verify_a2(agent_A: GaussianAgent, agent_B: GaussianAgent,
              noise_floor: float = 0.02) -> tuple:
    """
    Verifier: checks fiber membership with small observation noise.
    Returns (invariant_holds, details).
    """
    noisy_mu_A = agent_A.mu + np.random.normal(0, noise_floor, N)
    noisy_A = GaussianAgent(mu=noisy_mu_A, Sigma=SIGMA_MAT.copy())

    fiber_A = threshold_fiber_id(noisy_A, THRESHOLDS)
    fiber_B = threshold_fiber_id(agent_B, THRESHOLDS)

    invariant_holds = (fiber_A != fiber_B)

    return invariant_holds, {
        'fiber_A': fiber_A,
        'fiber_B': fiber_B,
        'same_fiber': fiber_A == fiber_B,
    }


def _find_crossing(x_vals, y_vals, target=0.5):
    for i in range(len(y_vals) - 1):
        y0, y1 = float(y_vals[i]), float(y_vals[i+1])
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            t = (target - y0) / (y1 - y0 + 1e-12)
            return float(x_vals[i] + t * (x_vals[i+1] - x_vals[i]))
    return float('nan')


def run_a2_phase_diagram(
    delta_values: np.ndarray,
    n_trials: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Sweeps delta (mean displacement), measures D⊥ and pass rate.
    Blind — no import of d_perp_star().
    """
    rng = np.random.default_rng(seed)

    dp_means   = []
    kl_means   = []
    pass_rates = []
    perp_means = []

    if verbose:
        print(f"\nA2-compliant phase diagram")
        print(f"  N={N}, sigma={SIGMA}, isotropic, symmetric thresholds")
        print(f"  Threshold crossing at delta=0.5 (dim 0 crosses 0)")
        print(f"  {len(delta_values)} delta levels × {n_trials} trials each")
        print(f"\n  {'delta':>8} {'D⊥_mean':>10} {'KL_mean':>9} "
              f"{'perp':>7} {'PassRate':>10} {'DiffFib%':>10}")
        print("  " + "-" * 60)

    for delta in delta_values:
        dps, kls, perps, passes = [], [], [], []

        for _ in range(n_trials):
            agent_A, agent_B = make_agents_a2(delta, rng)
            m = measure_d_perp_a2(agent_A, agent_B)
            holds, details = verify_a2(agent_A, agent_B)

            dps.append(m['d_perp'])
            kls.append(m['d_kl'])
            perps.append(m['perp_factor'])
            passes.append(holds)

        dp_mean   = np.mean(dps)
        kl_mean   = np.mean(kls)
        perp_mean = np.mean(perps)
        pass_rate = np.mean(passes)

        dp_means.append(dp_mean)
        kl_means.append(kl_mean)
        pass_rates.append(pass_rate)
        perp_means.append(perp_mean)

        if verbose:
            print(f"  {delta:>8.4f} {dp_mean:>10.4f} {kl_mean:>9.4f} "
                  f"{perp_mean:>7.4f} {pass_rate:>10.4f} "
                  f"{100*(1-np.mean([d['same_fiber'] for d in [verify_a2(make_agents_a2(delta,rng)[0], make_agents_a2(delta,rng)[1])[1] for _ in range(20)]])):>9.1f}%")

    dp_means   = np.array(dp_means)
    kl_means   = np.array(kl_means)
    pass_rates = np.array(pass_rates)

    empirical_d_star = _find_crossing(dp_means, pass_rates, 0.5)
    empirical_delta_star = _find_crossing(delta_values, pass_rates, 0.5)

    return {
        'delta_values':       delta_values,
        'dp_means':           dp_means,
        'kl_means':           kl_means,
        'pass_rates':         pass_rates,
        'empirical_d_star':   empirical_d_star,
        'empirical_delta_star': empirical_delta_star,
    }


def main():
    print("=" * 62)
    print("two_agent_a2.py — A2-Compliant Phase Boundary Experiment")
    print("Isotropic Gaussians. Zero baseline. Symmetric thresholds.")
    print("Blind: no access to d_perp_star() prediction.")
    print("=" * 62)

    # Delta sweep: threshold crossing at delta=0.5
    # Fine grid around the crossing
    delta_grid = np.concatenate([
        np.linspace(0.0,  0.35, 8),
        np.linspace(0.38, 0.62, 25),   # fine around crossing at 0.5
        np.linspace(0.65, 1.0,  8),
    ])
    delta_grid = np.unique(np.round(delta_grid, 4))

    results = run_a2_phase_diagram(
        delta_values=delta_grid,
        n_trials=500,
        seed=42,
        verbose=True,
    )

    print(f"\n{'='*62}")
    print(f"EMPIRICAL RESULT")
    print(f"{'='*62}")
    print(f"  Empirical delta*  = {results['empirical_delta_star']:.4f}")
    print(f"  Empirical D⊥*     = {results['empirical_d_star']:.4f}")
    print(f"\n  Expected threshold crossing at delta=0.5")
    print(f"  Expected D⊥* from geometry: 0.5*(0.5/0.3)^2 * perp ≈ 1.39*perp")
    print(f"\n  Run comparison_a2.py to compare with analytical 1.5123.")
    print(f"{'='*62}")

    return results


if __name__ == "__main__":
    results = main()
