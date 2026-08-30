# OPD — Orbital Propellant Depot (lumped-parameter model)

Python package for transient simulation of cryogenic hydrogen tanks with optional
adsorbent modules. Fluid properties come from
[CoolProp](https://coolprop.org); adsorption follows Dubinin–Astakhov /
hybrid sub-/supercritical models.

**Units:** SI throughout (K, Pa, mol, kg, J, s).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Architecture

```
opd/
├── fluids/          CoolProp wrapper (parahydrogen default)
├── adsorbents/      208C activated carbon, AX-21, MOF presets
├── tank/            geometry, heat loads, Tank / TwoTempTank ODEs
├── cryocooler/      constant and Carnot-limited models
├── control/         pressure controllers (bang-bang, proportional, …)
├── catalysts/       para–ortho conversion
├── environment/     orbital / MLI heat-flux helpers
└── simulation/      EOS resolver, TransientSimulator, results
```

Dependency direction: `fluids` / `adsorbents` → `tank` → `simulation`.

## Quick example

See [`examples/01_bare_tank_transient.ipynb`](examples/01_bare_tank_transient.ipynb)
(or run the same logic as a script: `python examples/bare_tank_transient.py`).

The notebook integrates a 1 m³ tank with a constant heat leak, plots pressure
and temperature vs. time, and compares bare vs. activated-carbon loading.

## Adsorbents included

| Key | Material | Notes |
|-----|----------|-------|
| `activated_carbon` | 208C (NZ208V3) | Cryogenic isotherm fits, hybrid D-A |
| `ax21` | AX-21 | Extended warm-dense range |
| `MIL101` | MIL-101(Cr) | Literature MOF parameters |
| `IRMOF20` | IRMOF-20 | Literature MOF parameters |

```python
from opd.adsorbents import get_adsorbent
ac = get_adsorbent("activated_carbon")
```

## Citation

If you use this code in academic work, please cite the associated conference
paper (details in publication list). This repository contains the simulation
framework only; it does not bundle manuscripts or measurement raw data.

## License

MIT — see [LICENSE](LICENSE).
