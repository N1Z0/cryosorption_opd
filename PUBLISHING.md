# Publishing OPD on GitHub

This folder holds templates for a **public** release. Your working tree under
`OPD/` may contain papers, raw measurement paths, and study scripts that must
**not** be pushed.

## What gets published

| Included | Excluded (stay local) |
|----------|------------------------|
| `opd/` package | `docs/paper*`, `docs/poster`, `docs/theory_manual.tex` |
| `tests/` | `results/` |
| `examples/` | `run_warm_dense_study.py` (local CSV paths) |
| `README.md`, `LICENSE`, `pyproject.toml` | `run_business_case.py`, `viz_pro.py`, `scenarios/` |
| | `.cursorrules`, `.cursor/`, `.venv/` |

## Build the release folder

```bash
chmod +x release/build_public.sh
./release/build_public.sh
```

This creates `../OPD-public/` (override destination: `./release/build_public.sh /path/to/dir`).

Verify:

```bash
cd ../OPD-public
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/bare_tank_transient.py
```

## Push to GitHub

```bash
cd ../OPD-public
git init
git add .
git commit -m "Initial public release of OPD lumped-parameter framework"
```

Create an empty repository on GitHub (e.g. `your-org/opd`), then:

```bash
git remote add origin git@github.com:YOUR_USER/opd.git
git branch -M main
git push -u origin main
```

## Before you push — checklist

- [ ] No files under `results/` or `docs/paper*`
- [ ] Grep for personal paths: `rg '/Users/' .` should return nothing
- [ ] Grep for patent numbers if you want them out of code comments: `rg '25168547|636,249' .`
- [ ] Update `LICENSE` copyright line if needed
- [ ] Add a GitHub repo description and link to your paper separately

## Rebuilding after code changes

Re-run `./release/build_public.sh` from the main `OPD/` tree whenever you update
the library. Commit and push from `OPD-public/` only.
