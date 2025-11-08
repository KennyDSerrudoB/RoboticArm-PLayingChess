# Pc/common/config_loader.py
# ------------------------------------------------------------
# CARGADOR DE CONFIGURACIÓN CENTRAL DEL PROYECTO
#
# Objetivo:
#   - Leer el archivo YAML de configuración del proyecto (Data/config.yaml).
#   - (Opcional) Hacer overlay con Data/config.local.yaml para ajustes personales.
#   - Convertir rutas relativas en absolutas (portabilidad).
#   - Crear directorios base si no existen (Data, Logs, etc.).
#   - Exponer funciones cómodas para obtener la configuración y rutas.
#
# Requisitos:
#   pip install pyyaml
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

# ============================================================
# 1) HELPERS INTERNOS (funciones auxiliares)
# ============================================================

def _project_root() -> Path:
    """
    Devuelve la raíz del repositorio/proyecto.
    - Toma la ruta absoluta de ESTE archivo (Pc/common/config_loader.py).
    - Sube 2 niveles de carpetas: .../Pc/common/ -> .../Pc/ -> .../<RAIZ>.
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Lee un archivo YAML y devuelve un dict de Python.
    - Si el archivo está vacío, devuelve {}.
    - Si el contenido raíz del YAML NO es un dict, lanza ValueError.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict). Revisa: {path}")
    return data


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any] | None) -> None:
    """
    Merge recursivo: inserta las claves de `upd` dentro de `base` sin
    destruir subdiccionarios existentes.
    """
    if not upd:
        return
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _normalize_paths_recursive(node: Any, root: Path) -> Any:
    """
    Normaliza rutas de manera RECURSIVA en cualquier estructura dentro de `paths`.
    - Si encuentra strings que parecen rutas relativas, las convierte a absolutas con `root`.
    - Mantiene tipos de listas/dicts anidados.
    - Si encuentra valores no-string, los deja como están.
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out[k] = _normalize_paths_recursive(v, root)
        return out
    elif isinstance(node, list):
        return [_normalize_paths_recursive(v, root) for v in node]
    elif isinstance(node, str):
        p = Path(node)
        if p.is_absolute():
            return str(p)
        # Consideramos string como ruta si contiene separadores o parece archivo
        if any(sep in node for sep in ("/", "\\")) or "." in Path(node).name:
            return str((root / p).resolve())
        return node
    else:
        return node


def _normalize_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """
    Aplica normalización recursiva SOLO dentro de cfg['paths'] si existe.
    """
    raw = cfg.get("paths", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("La sección 'paths' debe ser un diccionario.")
    return _normalize_paths_recursive(raw, root)


def _ensure_base_dirs(paths: Dict[str, Any]) -> None:
    """
    Crea directorios base si están definidos en 'paths'.
    - No crea archivos, SOLO carpetas típicas.
    - Incluye 'snapshots_dir' para visión.
    """
    for key in ("data_dir", "datasets_dir", "logs_dir", "debug_dir", "snapshots_dir"):
        p = paths.get(key)
        if isinstance(p, str) and p:
            Path(p).mkdir(parents=True, exist_ok=True)

# ============================================================
# 2) API PÚBLICA
# ============================================================

_CFG_CACHE: Dict[str, Any] | None = None

def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Carga la configuración del proyecto con este orden de prioridad:
      1) Data/config.yaml        (principal)
      2) Data/configInit.yaml    (fallback si no existe el principal)
      3) Data/config.local.yaml  (overlay opcional; no se versiona)

    Luego:
      - Normaliza recursivamente las rutas en cfg['paths'] a absolutas.
      - Crea directorios base si están definidos (data_dir, datasets_dir, logs_dir, debug_dir, snapshots_dir).
      - Adjunta metadatos útiles en cfg["_meta"].

    `force_reload=True` invalida la caché en memoria.
    """
    global _CFG_CACHE
    if _CFG_CACHE is not None and not force_reload:
        return _CFG_CACHE

    root = _project_root()
    data_dir = root / "Data"

    primary   = data_dir / "config.yaml"
    fallback  = data_dir / "configInit.yaml"
    localfile = data_dir / "config.local.yaml"

    # 1) Archivo base
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

    # 2) Overlay local (opcional)
    if localfile.exists():
        _deep_update(cfg, _load_yaml(localfile))

    # 3) Normalizar rutas dentro de 'paths' (recursivo) y crear dir base
    paths_abs = _normalize_paths(cfg, root)
    cfg["paths"] = paths_abs
    _ensure_base_dirs(paths_abs)

    # 4) Metadatos
    cfg["_meta"] = {
        "project_root": str(root),
        "cfg_source": str(source),
        "cfg_local": str(localfile) if localfile.exists() else None,
    }

    _CFG_CACHE = cfg
    return cfg


def get_path(key: str) -> Path:
    """
    Devuelve una ruta absoluta de cfg['paths'][key] como Path.
    """
    cfg = load_config()
    try:
        return Path(cfg["paths"][key])
    except KeyError as e:
        raise KeyError(f"No existe 'paths.{key}' en la configuración.") from e


def get_camera_config() -> Dict[str, Any]:
    """
    Devuelve un dict (copia) con la sección 'camera' del YAML.
    """
    cfg = load_config()
    return (cfg.get("camera") or {}).copy()


def get_engine_config() -> Dict[str, Any]:
    """
    Devuelve un dict (copia) con la sección 'engine' del YAML.
    """
    cfg = load_config()
    return (cfg.get("engine") or {}).copy()


def project_root() -> Path:
    """Atajo para obtener la raíz del proyecto como Path."""
    return _project_root()
