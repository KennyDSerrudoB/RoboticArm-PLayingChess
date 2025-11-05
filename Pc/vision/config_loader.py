import os, yaml, pathlib

def _deep_update(b, u):
    for k,v in (u or {}).items():
        if isinstance(v, dict) and isinstance(b.get(k), dict):
            _deep_update(b[k], v)
        else:
            b[k] = v

def load_config():
    root = pathlib.Path(__file__).resolve().parents[2]
    cfg_path = root / "Data" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # override local opcional
    loc = root / "Data" / "config.local.yaml"
    if loc.exists():
        with open(loc, "r") as f:
            _deep_update(cfg, yaml.safe_load(f))

    # rutas absolutas
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        if isinstance(v, str) and not os.path.isabs(v):
            paths[k] = str((root / v).resolve())

    cfg["paths"] = paths
    return cfg