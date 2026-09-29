import shutil
from pathlib import Path

import pytest

from lab.cli import write_dry_workspace
from lab.config import load_config
from lab.run import Run


@pytest.fixture()
def make_run(tmp_path):
    """Fresh mock run in its own workspace. Extra config keys can be merged in per test."""
    counter = {"n": 0}

    def _make(**overrides):
        counter["n"] += 1
        root = tmp_path / f"ws{counter['n']}"
        cfgp = write_dry_workspace(root)
        if overrides:
            import yaml

            d = yaml.safe_load(cfgp.read_text())
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    d[k].update(v)
                else:
                    d[k] = v
            cfgp.write_text(yaml.safe_dump(d))
        cfg = load_config(cfgp)
        return Run(cfg, root / "runs" / cfg.run_id)

    return _make
