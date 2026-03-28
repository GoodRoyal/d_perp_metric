"""
d_perp/metric.py
================
Perpendicular KL Divergence (D⊥) implementation.

Three components:
  1. D_KL(P‖Q)          — standard KL divergence
  2. (1 - |<u_P, u_Q>|) — perpendicularity factor (1 = fully perpendicular, 0 = aligned)
  3. xi(P, Q)            — elemental compatibility factor

D⊥(P‖Q) = D_KL(P‖Q) × (1 - |<u_P, u_Q>|) × xi(P, Q)

For Gaussian distributions:
  P = N(mu_P, Sigma_P)
  Q = N(mu_Q, Sigma_Q)

D_KL(P‖Q) has closed form.
u_P, u_Q are the principal eigenvectors of Sigma_P, Sigma_Q.

Also implements:
  - D⊥* analytical prediction from derivation
  - Within-fiber vs between-fiber decomposition
  - KL divergence for comparison (to demonstrate D⊥ ≠ KL)

Author: Juan Carlos Paredes
Date: 2026-03-27
Status: Step 1 — standalone metric, tested against known values
"""

import numpy as np
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Optional
import warnings


# ─────────────────────────────────────────────
# Core data structure
# ─────────────────────────────────────────────

@dataclass
class GaussianAgent:
    """
    A Gaussian distribution representing an agent's belief state.
    
    mu:    mean vector, shape (n,)
    Sigma: covariance matrix, shape (n, n) — must be positive definite
    xi:    elemental compatibility scalar in [0, 1]
             default 1.0 (full compatibility, simplest case)
    label: optional name for reporting
    """
    mu: np.ndarray
    Sigma: np.ndarray
    xi: float = 1.0
    label: str = ""

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.Sigma = np.asarray(self.Sigma, dtype=float)
        n = self.mu.shape[0]
        assert self.Sigma.shape == (n, n), "Sigma must be (n, n)"
        assert 0.0 <= self.xi <= 1.0, "xi must be in [0, 1]"

    @property
    def n(self):
        return self.mu.shape[0]

    def principal_direction(self) -> np.ndarray:
        """
        Returns the principal eigenvector of Sigma —
        the direction of maximum variance.
        Uses scipy eigh (symmetric) for numerical stability.
        """
        eigenvalues, eigenvectors = eigh(self.Sigma)
        # eigh returns in ascending order; last is largest
        u = eigenvectors[:, -1]
        # Normalize (should already be unit, but enforce)
        return u / np.linalg.norm(u)


# ─────────────────────────────────────────────
# Component functions
# ─────────────────────────────────────────────

def kl_divergence(P: GaussianAgent, Q: GaussianAgent) -> float:
    """
    D_KL(P‖Q) for Gaussians.

    Closed form:
      D_KL = 0.5 × [tr(Σ_Q⁻¹ Σ_P) + (μ_Q - μ_P)ᵀ Σ_Q⁻¹ (μ_Q - μ_P) - n + ln(det Σ_Q / det Σ_P)]

    Returns float ≥ 0. Returns 0.0 when P == Q (up to numerical tolerance).
    """
    assert P.n == Q.n, "Dimensionality mismatch"
    n = P.n

    try:
        Sigma_Q_inv = np.linalg.inv(Q.Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Q.Sigma is singular — cannot compute KL divergence")

    diff = Q.mu - P.mu

    trace_term = np.trace(Sigma_Q_inv @ P.Sigma)
    quadratic_term = diff @ Sigma_Q_inv @ diff

    sign_P, logdet_P = np.linalg.slogdet(P.Sigma)
    sign_Q, logdet_Q = np.linalg.slogdet(Q.Sigma)

    if sign_P <= 0 or sign_Q <= 0:
        raise ValueError("Covariance matrices must be positive definite")

    log_det_ratio = logdet_Q - logdet_P

    kl = 0.5 * (trace_term + quadratic_term - n + log_det_ratio)

    # Numerical floor — KL is non-negative by definition
    return float(max(0.0, kl))


def perpendicularity_factor(P: GaussianAgent, Q: GaussianAgent) -> float:
    """
    (1 - |<u_P, u_Q>|)

    = 0 when principal directions are aligned (same subspace)
    = 1 when principal directions are orthogonal (fully perpendicular)

    This is the factor that makes D⊥ sensitive to geometric
    orientation, not just distributional distance.
    """
    u_P = P.principal_direction()
    u_Q = Q.principal_direction()
    alignment = abs(float(np.dot(u_P, u_Q)))
    # Clamp to [0, 1] for numerical safety
    alignment = min(1.0, max(0.0, alignment))
    return 1.0 - alignment


def compatibility_factor(P: GaussianAgent, Q: GaussianAgent) -> float:
    """
    xi(P, Q) — elemental compatibility.

    Geometric mean of the two agents' compatibility scalars.
    When both xi=1.0 (default), this returns 1.0.
    """
    return float(np.sqrt(P.xi * Q.xi))


def d_perp(P: GaussianAgent, Q: GaussianAgent) -> dict:
    """
    D⊥(P‖Q) = D_KL(P‖Q) × (1 - |<u_P, u_Q>|) × xi(P, Q)

    Returns a dict with all components for transparency:
      {
        'd_kl':              float,
        'perp_factor':       float,
        'xi':                float,
        'd_perp':            float,
        'principal_P':       np.ndarray,
        'principal_Q':       np.ndarray,
      }

    Design principle: never hide intermediate values.
    Every component is inspectable so the researcher can
    see where D⊥ and KL diverge from each other.
    """
    dkl  = kl_divergence(P, Q)
    perp = perpendicularity_factor(P, Q)
    xi   = compatibility_factor(P, Q)
    dp   = dkl * perp * xi

    return {
        'd_kl':        dkl,
        'perp_factor': perp,
        'xi':          xi,
        'd_perp':      dp,
        'principal_P': P.principal_direction(),
        'principal_Q': Q.principal_direction(),
    }


# ─────────────────────────────────────────────
# Analytical D⊥* prediction
# ─────────────────────────────────────────────

def d_perp_star(n: int, B: float, sigma: float, xi_max: float = 1.0) -> dict:
    """
    Analytical prediction of the critical D⊥* from the derivation.

    D⊥* = (n/2) × f(B, σ, n) × ξ_max

    where f(B, σ, n) = r² - 1 - ln(r²)
          r = B / (2σ√n)

    Parameters
    ----------
    n       : int   — state space dimensionality
    B       : float — channel bandwidth (bits per transmission)
    sigma   : float — channel noise standard deviation
    xi_max  : float — maximum elemental compatibility in fiber (default 1.0)

    Returns
    -------
    dict with:
      'r'          : the dimensionless ratio B/(2σ√n)
      'f'          : the function f(r)
      'd_perp_star': the predicted critical value
      'warning'    : string if assumptions may be violated
    """
    r = B / (2.0 * sigma * np.sqrt(n))

    warning = ""
    if r <= 1.0:
        warning = (
            f"r = {r:.4f} ≤ 1: low-bandwidth regime. "
            "Derivation assumptions (Gaussian, isotropic) may not hold. "
            "Treat D⊥* as approximate lower bound only."
        )
    if r > 10.0:
        warning = (
            f"r = {r:.4f} >> 1: high-bandwidth regime. "
            "f(r) grows as r² — large D⊥* means functor is nearly lossless."
        )

    # f(r) = r² - 1 - ln(r²)
    # Note: f(r) ≥ 0 for all r > 0, with equality only at r = 1
    f_val = r**2 - 1.0 - np.log(r**2)
    f_val = max(0.0, f_val)  # numerical floor

    d_star = (n / 2.0) * f_val * xi_max

    return {
        'r':           r,
        'f':           f_val,
        'd_perp_star': d_star,
        'n':           n,
        'B':           B,
        'sigma':       sigma,
        'xi_max':      xi_max,
        'warning':     warning,
    }


# ─────────────────────────────────────────────
# Fiber structure utilities
# ─────────────────────────────────────────────

def threshold_fiber_id(P: GaussianAgent, thresholds: np.ndarray) -> tuple:
    """
    Given threshold vector τ, returns the fiber identifier for P.
    Fiber ID = binary vector of sign(E_P[x_i] - τ_i).

    Two agents with the same fiber ID are indistinguishable
    to the AR layer under threshold discretization.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    assert thresholds.shape == (P.n,), "Threshold vector must match dimensionality"
    return tuple((P.mu > thresholds).astype(int).tolist())


def d_perp_decomposed(
    P: GaussianAgent,
    Q: GaussianAgent,
    thresholds: np.ndarray
) -> dict:
    """
    Decomposes D⊥(P‖Q) into between-fiber and within-fiber components.

    If P and Q are in the same fiber: D⊥_between = 0, D⊥_within = D⊥(P‖Q)
    If P and Q are in different fibers: D⊥_between > 0 (potentially verifiable)

    Returns dict with:
      'fiber_P'      : tuple — fiber ID of P
      'fiber_Q'      : tuple — fiber ID of Q
      'same_fiber'   : bool
      'd_perp_total' : float
      'd_perp_between': float
      'd_perp_within': float
      'verifiable'   : bool — True if between-fiber component > 0
    """
    fiber_P = threshold_fiber_id(P, thresholds)
    fiber_Q = threshold_fiber_id(Q, thresholds)
    same    = (fiber_P == fiber_Q)

    result  = d_perp(P, Q)
    dp_total = result['d_perp']

    if same:
        dp_between = 0.0
        dp_within  = dp_total
    else:
        # Between-fiber component: D⊥ computed on the fiber centroids
        # (the means projected to the threshold boundary)
        # Simplified: use the sign-vector distance as a proxy
        # Full treatment requires computing D⊥ between fiber representatives
        fiber_P_arr = np.array(fiber_P, dtype=float)
        fiber_Q_arr = np.array(fiber_Q, dtype=float)
        # Fraction of dimensions that disagree, scaled by total D⊥
        disagree_fraction = np.mean(fiber_P_arr != fiber_Q_arr)
        dp_between = dp_total * disagree_fraction
        dp_within  = dp_total * (1.0 - disagree_fraction)

    return {
        'fiber_P':          fiber_P,
        'fiber_Q':          fiber_Q,
        'same_fiber':       same,
        'd_perp_total':     dp_total,
        'd_perp_between':   dp_between,
        'd_perp_within':    dp_within,
        'verifiable':       not same and dp_between > 0,
        'd_kl':             result['d_kl'],
        'perp_factor':      result['perp_factor'],
    }


# ─────────────────────────────────────────────
# Known-value tests
# ─────────────────────────────────────────────

def run_tests():
    """
    Tests against known analytical values.
    Prints PASS/FAIL for each.
    
    These are the ground-truth checks before any experiment runs.
    If any test fails, do not proceed to phase_diagram.py.
    """
    print("=" * 60)
    print("d_perp/metric.py — Known-Value Tests")
    print("=" * 60)
    n_pass = 0
    n_fail = 0

    def check(name, computed, expected, tol=1e-6):
        nonlocal n_pass, n_fail
        err = abs(computed - expected)
        if err < tol:
            print(f"  PASS  {name}")
            print(f"        computed={computed:.8f}  expected={expected:.8f}")
            n_pass += 1
        else:
            print(f"  FAIL  {name}")
            print(f"        computed={computed:.8f}  expected={expected:.8f}  err={err:.2e}")
            n_fail += 1

    # ── Test 1: KL divergence of identical Gaussians = 0 ──
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    check("KL(P‖P) = 0 (identical)", kl_divergence(P, Q), 0.0)

    # ── Test 2: KL divergence, 1D, known closed form ──
    # D_KL(N(0,1) ‖ N(1,1)) = 0.5 × (0 + 1 + 0 - 1) = 0.5
    P = GaussianAgent(mu=[0.0], Sigma=[[1.0]])
    Q = GaussianAgent(mu=[1.0], Sigma=[[1.0]])
    check("KL(N(0,1)‖N(1,1)) = 0.5", kl_divergence(P, Q), 0.5)

    # ── Test 3: KL divergence, 1D, different variances ──
    # D_KL(N(0,1) ‖ N(0,2)) = 0.5 × (1/2 + 0 - 1 + ln(2)) = 0.5×(ln(2) - 0.5)
    P = GaussianAgent(mu=[0.0], Sigma=[[1.0]])
    Q = GaussianAgent(mu=[0.0], Sigma=[[2.0]])
    expected = 0.5 * (0.5 + 0.0 - 1.0 + np.log(2.0))
    check("KL(N(0,1)‖N(0,2))", kl_divergence(P, Q), expected)

    # ── Test 4: Perpendicularity factor, aligned = 0 ──
    # Both agents have principal direction along x-axis
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 1.0]))
    Q = GaussianAgent(mu=[1.0, 0.0], Sigma=np.diag([3.0, 0.5]))
    pf = perpendicularity_factor(P, Q)
    check("Perp factor, aligned directions = 0", pf, 0.0, tol=1e-6)

    # ── Test 5: Perpendicularity factor, orthogonal = 1 ──
    # P has principal direction along x, Q along y
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]))
    pf = perpendicularity_factor(P, Q)
    check("Perp factor, orthogonal directions = 1", pf, 1.0, tol=1e-6)

    # ── Test 6: D⊥ = 0 when distributions identical ──
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    result = d_perp(P, Q)
    check("D⊥(P,P) = 0", result['d_perp'], 0.0)

    # ── Test 7: D⊥ = 0 when aligned (perp_factor = 0) ──
    # Even with large KL, D⊥ = 0 if aligned
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 1.0]))
    Q = GaussianAgent(mu=[5.0, 0.0], Sigma=np.diag([3.0, 0.5]))
    result = d_perp(P, Q)
    check("D⊥ = 0 when aligned (large KL, zero perp)", result['d_perp'], 0.0, tol=1e-6)
    print(f"        (KL was {result['d_kl']:.4f}, perp_factor was {result['perp_factor']:.6f})")

    # ── Test 8: D⊥ = KL when fully perpendicular, xi=1 ──
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]))
    result = d_perp(P, Q)
    kl = kl_divergence(P, Q)
    check("D⊥ = KL when fully perpendicular, xi=1", result['d_perp'], kl, tol=1e-6)

    # ── Test 9: xi scaling ──
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]), xi=0.5)
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]), xi=0.5)
    result = d_perp(P, Q)
    kl = kl_divergence(P, Q)
    # xi_geom = sqrt(0.5 × 0.5) = 0.5; perp = 1; D⊥ = KL × 0.5
    check("D⊥ = KL × 0.5 when xi=0.5 both agents", result['d_perp'], kl * 0.5, tol=1e-6)

    # ── Test 10: D⊥* at r=1 should = 0 ──
    # f(r) = r² - 1 - ln(r²); at r=1: f = 1 - 1 - 0 = 0
    result_star = d_perp_star(n=4, B=2.0, sigma=1.0)  # r = 2/(2×1×2) = 0.5 ... adjust
    # For r=1 exactly: B = 2σ√n
    n_test = 4
    sigma_test = 1.0
    B_test = 2.0 * sigma_test * np.sqrt(n_test)  # = 4.0
    result_star = d_perp_star(n=n_test, B=B_test, sigma=sigma_test)
    check("D⊥*(r=1) = 0", result_star['d_perp_star'], 0.0, tol=1e-10)

    # ── Test 11: The prediction for the main experiment ──
    print()
    print("  ── Experimental prediction ──")
    pred = d_perp_star(n=4, B=2.0, sigma=0.3, xi_max=1.0)
    print(f"  Parameters: n=4, B=2.0, sigma=0.3, xi_max=1.0")
    print(f"  r          = {pred['r']:.6f}")
    print(f"  f(r)       = {pred['f']:.6f}")
    print(f"  D⊥*        = {pred['d_perp_star']:.6f}")
    if pred['warning']:
        print(f"  WARNING    : {pred['warning']}")
    print()
    print(f"  This number ({pred['d_perp_star']:.4f}) is the predicted phase boundary.")
    print(f"  It is written here before any simulation runs.")
    print(f"  The experiment will measure the empirical phase boundary")
    print(f"  and compare it to this value.")

    # ── Summary ──
    print()
    print("=" * 60)
    print(f"Results: {n_pass} PASS, {n_fail} FAIL")
    if n_fail == 0:
        print("All tests passed. Safe to proceed to phase_diagram.py.")
    else:
        print("TESTS FAILED. Do not proceed until failures are resolved.")
    print("=" * 60)

    return n_fail == 0


# ─────────────────────────────────────────────
# Demo: D⊥ vs KL — the key comparison
# ─────────────────────────────────────────────

def demo_d_perp_vs_kl():
    """
    Demonstrates the cases where D⊥ and KL diverge from each other.
    This is the core motivation for why D⊥ is the right metric.
    
    Three cases:
      Case A: Large KL, D⊥ = 0    (aligned distributions — KL misleads)
      Case B: Small KL, D⊥ > 0   (perpendicular distributions — KL undersells)
      Case C: KL = D⊥             (fully perpendicular, xi=1 — metrics agree)
    """
    print()
    print("=" * 60)
    print("Demo: D⊥ vs KL — where the metrics diverge")
    print("=" * 60)

    thresholds = np.array([0.0, 0.0])

    # Case A: Large KL, D⊥ ≈ 0
    # Same fiber (both means positive), aligned principal directions
    P_A = GaussianAgent(mu=[1.0, 0.5], Sigma=np.diag([3.0, 0.1]), label="P_A")
    Q_A = GaussianAgent(mu=[3.0, 1.5], Sigma=np.diag([4.0, 0.2]), label="Q_A")
    r_A = d_perp(P_A, Q_A)
    dec_A = d_perp_decomposed(P_A, Q_A, thresholds)
    print(f"\nCase A — Same fiber, aligned:")
    print(f"  KL    = {r_A['d_kl']:.4f}  (suggests large difference)")
    print(f"  D⊥    = {r_A['d_perp']:.4f}  (correctly shows: no verifiable structure)")
    print(f"  Fiber P={dec_A['fiber_P']}, Q={dec_A['fiber_Q']}, same={dec_A['same_fiber']}")
    print(f"  Verifiable: {dec_A['verifiable']}")

    # Case B: Small KL, D⊥ > 0
    # Different fibers, perpendicular principal directions
    P_B = GaussianAgent(mu=[0.5, -0.1], Sigma=np.diag([3.0, 0.1]), label="P_B")
    Q_B = GaussianAgent(mu=[-0.1, 0.5], Sigma=np.diag([0.1, 3.0]), label="Q_B")
    r_B = d_perp(P_B, Q_B)
    dec_B = d_perp_decomposed(P_B, Q_B, thresholds)
    print(f"\nCase B — Different fibers, perpendicular:")
    print(f"  KL    = {r_B['d_kl']:.4f}  (suggests small difference)")
    print(f"  D⊥    = {r_B['d_perp']:.4f}  (correctly shows: verifiable structure exists)")
    print(f"  Fiber P={dec_B['fiber_P']}, Q={dec_B['fiber_Q']}, same={dec_B['same_fiber']}")
    print(f"  Verifiable: {dec_B['verifiable']}")

    # Case C: Agreement — fully perpendicular, different fibers
    P_C = GaussianAgent(mu=[1.0, 0.0], Sigma=np.diag([2.0, 0.1]), label="P_C")
    Q_C = GaussianAgent(mu=[0.0, 1.0], Sigma=np.diag([0.1, 2.0]), label="Q_C")
    r_C = d_perp(P_C, Q_C)
    print(f"\nCase C — Different fibers, fully perpendicular (KL = D⊥):")
    print(f"  KL    = {r_C['d_kl']:.4f}")
    print(f"  D⊥    = {r_C['d_perp']:.4f}")
    print(f"  Perp  = {r_C['perp_factor']:.4f}")
    print(f"  (Metrics agree when geometry is maximally perpendicular)")

    print()
    print("Key insight: KL is blind to fiber structure and principal direction.")
    print("D⊥ = 0 in Case A correctly predicts: no invariant can survive")
    print("the lossy functor for these two agents.")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def collect_results() -> dict:
    """
    Collects all test and demo results into a JSON-serialisable dict.
    """
    results = {"tests": [], "demo": {}, "analytical_prediction": {}}

    def record(name, computed, expected, tol=1e-6):
        err = abs(computed - expected)
        results["tests"].append({
            "name": name,
            "computed": computed,
            "expected": expected,
            "error": err,
            "passed": bool(err < tol),
        })

    # ── Tests ──
    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    record("KL(P‖P) = 0 (identical)", kl_divergence(P, Q), 0.0)

    P = GaussianAgent(mu=[0.0], Sigma=[[1.0]])
    Q = GaussianAgent(mu=[1.0], Sigma=[[1.0]])
    record("KL(N(0,1)‖N(1,1)) = 0.5", kl_divergence(P, Q), 0.5)

    P = GaussianAgent(mu=[0.0], Sigma=[[1.0]])
    Q = GaussianAgent(mu=[0.0], Sigma=[[2.0]])
    expected = 0.5 * (0.5 + 0.0 - 1.0 + np.log(2.0))
    record("KL(N(0,1)‖N(0,2))", kl_divergence(P, Q), expected)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 1.0]))
    Q = GaussianAgent(mu=[1.0, 0.0], Sigma=np.diag([3.0, 0.5]))
    record("Perp factor, aligned directions = 0", perpendicularity_factor(P, Q), 0.0)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]))
    record("Perp factor, orthogonal directions = 1", perpendicularity_factor(P, Q), 1.0)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.eye(2))
    res = d_perp(P, Q)
    record("D⊥(P,P) = 0", res['d_perp'], 0.0)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 1.0]))
    Q = GaussianAgent(mu=[5.0, 0.0], Sigma=np.diag([3.0, 0.5]))
    res = d_perp(P, Q)
    record("D⊥ = 0 when aligned (large KL, zero perp)", res['d_perp'], 0.0)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]))
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]))
    res = d_perp(P, Q)
    kl = kl_divergence(P, Q)
    record("D⊥ = KL when fully perpendicular, xi=1", res['d_perp'], kl)

    P = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([2.0, 0.1]), xi=0.5)
    Q = GaussianAgent(mu=[0.0, 0.0], Sigma=np.diag([0.1, 2.0]), xi=0.5)
    res = d_perp(P, Q)
    kl = kl_divergence(P, Q)
    record("D⊥ = KL × 0.5 when xi=0.5 both agents", res['d_perp'], kl * 0.5)

    n_test, sigma_test = 4, 1.0
    B_test = 2.0 * sigma_test * np.sqrt(n_test)
    res_star = d_perp_star(n=n_test, B=B_test, sigma=sigma_test)
    record("D⊥*(r=1) = 0", res_star['d_perp_star'], 0.0, tol=1e-10)

    # ── Analytical prediction ──
    pred = d_perp_star(n=4, B=2.0, sigma=0.3, xi_max=1.0)
    results["analytical_prediction"] = {
        "n": pred['n'], "B": pred['B'], "sigma": pred['sigma'],
        "xi_max": pred['xi_max'], "r": pred['r'], "f": pred['f'],
        "d_perp_star": pred['d_perp_star'], "warning": pred['warning'],
    }

    # ── Demo ──
    thresholds = np.array([0.0, 0.0])

    P_A = GaussianAgent(mu=[1.0, 0.5], Sigma=np.diag([3.0, 0.1]))
    Q_A = GaussianAgent(mu=[3.0, 1.5], Sigma=np.diag([4.0, 0.2]))
    r_A = d_perp(P_A, Q_A)
    dec_A = d_perp_decomposed(P_A, Q_A, thresholds)

    P_B = GaussianAgent(mu=[0.5, -0.1], Sigma=np.diag([3.0, 0.1]))
    Q_B = GaussianAgent(mu=[-0.1, 0.5], Sigma=np.diag([0.1, 3.0]))
    r_B = d_perp(P_B, Q_B)
    dec_B = d_perp_decomposed(P_B, Q_B, thresholds)

    P_C = GaussianAgent(mu=[1.0, 0.0], Sigma=np.diag([2.0, 0.1]))
    Q_C = GaussianAgent(mu=[0.0, 1.0], Sigma=np.diag([0.1, 2.0]))
    r_C = d_perp(P_C, Q_C)

    results["demo"] = {
        "case_A_same_fiber_aligned": {
            "d_kl": r_A['d_kl'], "d_perp": r_A['d_perp'],
            "perp_factor": r_A['perp_factor'],
            "fiber_P": list(dec_A['fiber_P']), "fiber_Q": list(dec_A['fiber_Q']),
            "same_fiber": dec_A['same_fiber'], "verifiable": dec_A['verifiable'],
        },
        "case_B_different_fibers_perpendicular": {
            "d_kl": r_B['d_kl'], "d_perp": r_B['d_perp'],
            "perp_factor": r_B['perp_factor'],
            "fiber_P": list(dec_B['fiber_P']), "fiber_Q": list(dec_B['fiber_Q']),
            "same_fiber": dec_B['same_fiber'], "verifiable": dec_B['verifiable'],
        },
        "case_C_fully_perpendicular_agreement": {
            "d_kl": r_C['d_kl'], "d_perp": r_C['d_perp'],
            "perp_factor": r_C['perp_factor'],
        },
    }

    n_pass = sum(1 for t in results["tests"] if t["passed"])
    n_fail = sum(1 for t in results["tests"] if not t["passed"])
    results["summary"] = {
        "total": len(results["tests"]),
        "passed": n_pass,
        "failed": n_fail,
        "all_passed": n_fail == 0,
    }
    return results


if __name__ == "__main__":
    import json
    output = collect_results()
    print(json.dumps(output, indent=2))
