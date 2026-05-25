# Contributing

Thanks for your interest! This repository is a research/engineering platform for
the **Enactic OpenArm v2** in MuJoCo. The notes below cover the developer workflow
and the conventions the codebase follows.

## Development setup

```bash
git clone <this-repo> && cd openarm_mujoco-master
python -m venv .venv && . .venv/bin/activate     # optional
pip install -e ".[all]"                          # core + rl + vision + dev tools
```

Verify the install:

```bash
openarm list           # all CLI commands
openarm test           # the headless test suite (≈150 tests)
```

## Running the tests

The full suite is **headless** (no display needed) and is the project's regression
gate — it must stay green.

```bash
python -m pytest tests/            # everything
python -m pytest tests/ -q -x      # stop at first failure
python -m pytest tests/test_openarm_bench.py -v
openarm test                       # same, via the CLI (passes args through)
```

## Code style

- **Formatter / linter:** [ruff](https://docs.astral.sh/ruff/) (`line-length = 100`,
  target `py310`). Run `ruff check .` and `ruff format .` before opening a PR.
- **Naming:** the package is `openarm_control` (never `control` — it collides with
  the `python-control` package). Match the surrounding code's style and comment density.
- **Imports:** keep heavy/optional deps (torch, ultralytics, mediapipe) **lazy**
  (import inside the function that needs them) so the core stays import-light.

## Conventions

- **Additive & non-breaking.** New skills go in new modules/scenes/tests; working
  controllers, scenes, and their tests are not edited. See
  [`docs/ROADMAP_EXTENSIONS.md`](docs/ROADMAP_EXTENSIONS.md) for how the extension
  arc was structured.
- **The OpenArm v2 model is never modified** — we only enrich the *world* it acts in
  and the *control/perception* around it.

### Adding a new skill (the usual pattern)

1. A controller module under `openarm_control/` (e.g. `articulated.py`).
2. A scene XML under `v2/openarm_mujoco_v2/`, registered in `openarm_control/config.py`.
3. A runnable demo under `openarm_control/demos/` with a `main(argv=None)` and a
   `--headless` self-test.
4. A CLI entry in `openarm_control/cli.py` (`COMMANDS`).
5. A headless test under `tests/`.
6. A line in [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md).

## Reporting issues & pull requests

Please open an issue on this repository's tracker for bugs and feature requests, and
PRs for patches. Review the [license](LICENSE) and [Code of Conduct](CODE_OF_CONDUCT.md)
first.

## Credit

The underlying OpenArm v2 MuJoCo model is by **Enactic, Inc.** — see
[docs.openarm.dev](https://docs.openarm.dev/simulation/mujoco) and their
[Discord](https://discord.gg/FsZaZ4z3We).
