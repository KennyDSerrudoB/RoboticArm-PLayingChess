# Pc/common/config_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

# ------------------------------------------------------------
# 1) Helpers internos
# ------------------------------------------------------------
def _project_root() -> Path:
    """
    Devuelve la raíz del proyecto.
    - Tomamos este archivo (Pc/common/config_loader.py)
    - Subimos 2 niveles -> raíz del repo.
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Lee un YAML y devuelve un dict (vacío si el archivo está vacío).
    Lanza error si el contenido NO es un mapeo clave→valor.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict). Revisa: {path}")
    return data


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any] | None) -> None:
    """
    Merge recursivo: inserta las claves de `upd` dentro de `base`,
    sin perder subdiccionarios.
    """
    if not upd:
        return
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _normalize_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, str]:
    """
    Convierte todas las rutas en cfg['paths'] a absolutas.
    - Si una ruta es relativa (p.ej. 'Data/archivo.txt'), la hace absoluta
      concatenando con `root` (la raíz del repo).
    - Devuelve un nuevo dict de rutas normalizadas (como strings).
    """
    raw = cfg.get("paths", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("La sección 'paths' debe ser un diccionario.")

    norm: Dict[str, str] = {}
    for key, val in raw.items():
        if not isinstance(val, str):
            # Permitimos solo strings en paths para evitar ambigüedades
            continue
        p = Path(val)
        if not p.is_absolute():
            p = (root / p).resolve()
        norm[key] = str(p)
    return norm


def _ensure_base_dirs(paths: Dict[str, str]) -> None:
    """
    Crea directorios base si están definidos.
    - No crea archivos, solo directorios típicos:
      'data_dir', 'datasets_dir', 'logs_dir', 'debug_dir'.
    """
    for key in ("data_dir", "datasets_dir", "logs_dir", "debug_dir"):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) API pública
# ------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """
    Carga la configuración del proyecto con este orden de prioridad:
    1) Data/config.yaml        (principal)
    2) Data/configInit.yaml    (fallback si no existe el principal)
    3) Data/config.local.yaml  (overlay opcional, no versionado)

    Luego:
    - Normaliza todas las rutas en cfg['paths'] a absolutas.
    - Crea los directorios base si están en paths (data_dir, datasets_dir, logs_dir, debug_dir).
    - Adjunta metadatos sobre qué archivo se usó.

    Retorna:
      dict con toda la configuración + cfg["_meta"] con información útil.
    """
    root = _project_root()
    data_dir = root / "Data"

    primary   = data_dir / "config.yaml"
    fallback  = data_dir / "configInit.yaml"
    localfile = data_dir / "config.local.yaml"

    if primary.exists():
        cfg = _load_yaml(primary)
        source = primary
    elif fallback.exists():
        cfg = _load_yaml(fallback)
        source = fallback
    else:
        raise FileNotFoundError(
            f"No se encontró {primary} ni {fallback}. "
            f"Crea uno de ellos dentro de {data_dir}."
        )

    # Overlay local (opcional): NO se versiona, ideal para tu PC
    if localfile.exists():
        _deep_update(cfg, _load_yaml(localfile))

    # Normalizar rutas (relativas -> absolutas) y crear dir base
    paths_abs = _normalize_paths(cfg, root)
    cfg["paths"] = paths_abs
    _ensure_base_dirs(paths_abs)

    # Metadatos útiles para debugging
    cfg["_meta"] = {
        "project_root": str(root),
        "cfg_source": str(source),
        "cfg_local": str(localfile) if localfile.exists() else None,
    }
    return cfg


def get_path(key: str) -> Path:
    """
    Acceso directo a una ruta de cfg['paths'][key] como Path.
    Ejemplo:
        get_path("data_dir") -> Path("/abs/ruta/al/proyecto/Data")
    """
    cfg = load_config()
    try:
        return Path(cfg["paths"][key])
    except KeyError as e:
        raise KeyError(f"No existe 'paths.{key}' en la configuración.") from e


def project_root() -> Path:
    """Atajo para obtener la raíz del proyecto como Path."""
    return _project_root()
