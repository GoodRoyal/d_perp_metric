# d_perp_metric

Research code for Perpendicular KL Divergence (D_perp) experiments, analytical derivations, and empirical checks under Assumption A2 and anisotropic extensions.

## Overview

This repository provides:

- A core D_perp metric implementation for Gaussian agents.
- Analytical D_perp* prediction utilities.
- Fiber-based decomposition and threshold crossing checks.
- Reproducible scripts that compare analytical predictions to empirical phase boundaries.

## Requirements

- Python 3.12+
- numpy
- scipy

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Files

- d_perp_metric.py: Core metric, KL utilities, D_perp* helper, fiber decomposition, and demo output.
- two_agent_a2.py: A2-compliant phase-boundary experiment.
- comparison_a2.py: Closes the loop between analytic and empirical A2 results.
- anisotropic_derivation.py: Derivation checks and anisotropic interpretation analysis.
- within_fiber_max.py: Within-fiber maximum analysis and constrained optimization discussion.

## Quick Start

Run each script directly:

```bash
source .venv/bin/activate
python d_perp_metric.py
python two_agent_a2.py
python comparison_a2.py
python anisotropic_derivation.py
python within_fiber_max.py
```

If you are running in a headless environment, use a non-interactive backend:

```bash
MPLBACKEND=Agg python two_agent_a2.py
```

## Validation Sweep

To run all scripts and check for failures:

```bash
for f in *.py; do
  echo "===== RUN $f ====="
  MPLBACKEND=Agg python "$f"
done
```

## Notes

- The scripts print explanatory analysis text to stdout and are intended for research validation, not as a packaged library.
- Current dependencies are intentionally minimal.

## License

This project is licensed under the MIT License.

See the LICENSE file for details.
