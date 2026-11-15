# Solver environment (S0)

One environment definition for every machine: `pyproject.toml` (extra `fe`) + `uv.lock` +
`.python-version` (3.12.12, exact patch: an unpinned "3.12" gave 3.12.12 on Bell but 3.12.13 on Negishi). Always sync with `--locked` so nobody silently re-resolves.

**Why Python is pinned to 3.12, not just ≥ 3.11:** `uv.lock` forks by interpreter — on 3.11
it resolves jax/jaxlib 0.10.2, on 3.12 jax 0.12.0.1. Same Python ⇒ same JAX everywhere.
(tatva 0.11.6's sparse tracer handles both the jax < 0.11 and ≥ 0.11 scan internals.)

Locked versions at S0 (commit 9962f73): Python 3.12.12, jax 0.12.0.1, tatva 0.11.6, uv 0.9.5.

## Two interpreters, on purpose

| Interpreter | Used for | Gets `tslid` via |
|---|---|---|
| uv env (`.venv`, Python 3.12) | `tslid.fe`, solver tests, analysis | editable install of the package |
| system `python3` (3.6.8) / `abaqus python` | `laws.py`, builders, ODB extractors | `sys.path` insert of `src/` |

`requires-python = ">=3.11"` in `pyproject.toml` only constrains the uv env; the ABAQUS side
never installs the package. `tslid/__init__.py` must not import `tslid.fe`
(guarded by `tests/fe/test_s0_smoke.py::test_tslid_laws_does_not_import_jax`).

## Install uv (once per cluster; Bell and Negishi have separate homes)

```bash
mkdir -p ~/.local/bin && cd /scratch/<cluster>/basu56 && mkdir -p .uv-dl && cd .uv-dl
curl -sSLf -o uv.tgz https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-musl.tar.gz
tar xzf uv.tgz && cp uv-x86_64-unknown-linux-musl/{uv,uvx} ~/.local/bin/
```

Keep uv's cache and managed Pythons on scratch, not in the home quota (put in `~/.bashrc`):

```bash
export UV_CACHE_DIR=/scratch/<cluster>/basu56/.uv-cache
export UV_PYTHON_INSTALL_DIR=/scratch/<cluster>/basu56/.uv-python
```

Laptop: install uv the usual way (`curl -LsSf https://astral.sh/uv/install.sh | sh`); defaults are fine.

## Sync and test

```bash
cd <repo root>
uv sync --locked --extra fe                     # downloads Python 3.12 on first use
uv run --locked --extra fe pytest tests/test_laws.py tests/fe -m "not slow"
python3 -m unittest discover -s tests -p "test_laws.py"     # old (ABAQUS-side) interpreter
```

Negishi compute node (from the Negishi clone root):

```bash
sbatch --account=tomar --partition=cpu scripts/slurm/run_fe_tests.sh            # "not slow"
sbatch --account=tomar --partition=cpu scripts/slurm/run_fe_tests.sh ""         # everything
```

Changing dependencies: edit `pyproject.toml`, run `uv lock`, commit `uv.lock` with it.

## Known hiccup (Bell login node, 2026-11-14)

The first `uv sync` failed with `Failed to query Python interpreter ... Resource temporarily
unavailable (os error 11)` right after uv downloaded Python. It was transient (re-run worked;
process and memory limits were far from exhausted). If it recurs: just re-run.

## GPU (not needed until S4)

- Negishi has partition `gpu`: nodes g000–g004, `Gres=gpu:3` each, `AllowAccounts=ALL` [D, `scontrol`, 2026-11-14].
  GPU model: **TBD**. Whether account `tomar` can actually run there (allocation / QOS): **TBD** — check with
  `sbatch --account=tomar --partition=gpu --gres=gpu:1 ...` or RCAC before S4.
- Other Purdue RCAC GPU clusters (e.g. Gilbreth, Gautschi) — not checked from here; access for this group: **TBD**.
- Plan for a GPU build (untested [H]): add an extra `fe-gpu = ["jax[cuda12]"]` (or `cuda13`, matching the
  node driver) to `pyproject.toml`, `uv lock`, then on a GPU node `uv sync --locked --extra fe --extra fe-gpu`
  and check `python -c "import jax; print(jax.devices())"`. Verify the float64 results agree with CPU.
