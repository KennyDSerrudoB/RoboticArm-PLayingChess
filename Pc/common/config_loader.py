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

from __future__ import annotations  # Mejora el manejo de "type hints" (opcional en Python >=3.11)
from pathlib import Path            # Manejo moderno de rutas (archivos y carpetas)
from typing import Any, Dict        # Tipos para anotar diccionarios y valores genéricos
import yaml                         # Lector/ escritor de archivos YAML (PyYAML)

# ============================================================
# 1) HELPERS INTERNOS (funciones auxiliares)
# ============================================================

def _project_root() -> Path:
    """
    Devuelve la raíz del repositorio/proyecto.
    ¿Cómo lo hace?
      - Toma la ruta absoluta de ESTE archivo (Pc/common/config_loader.py).
      - Sube 2 niveles de carpetas: .../Pc/common/ -> .../Pc/ -> .../<RAIZ>.
    ¿Por qué es útil?
      - Porque así podemos construir rutas robustas hacia 'Data/', 'Logs/', etc.,
        sin depender del 'directorio de trabajo actual' desde el que ejecutes Python.
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Lee un archivo YAML y devuelve un dict de Python.
    - Si el archivo está vacío, devuelve {}.
    - Si el contenido raíz del YAML NO es un dict (por ejemplo, es una lista),
      lanza un ValueError con un mensaje claro.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}  # safe_load: parsea datos, no ejecuta código
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict). Revisa: {path}")
    return data


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any] | None) -> None:
    """
    Merge (actualización) recursivo de diccionarios.
    - Inserta las claves/valores de 'upd' dentro de 'base' SIN destruir subdiccionarios existentes.
    - Se usa para aplicar 'config.local.yaml' por encima de 'config.yaml'.
    Ejemplo:
      base = {"a": {"x": 1, "y": 2}}
      upd  = {"a": {"y": 99}, "b": 3}
      => base queda {"a": {"x": 1, "y": 99}, "b": 3}
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
    Convierte todas las rutas en cfg['paths'] a ABSOLUTAS (strings).
    - Si una ruta está definida como relativa (p.ej., 'Data/archivo.txt'),
      la convierte a absoluta usando la raíz del proyecto 'root'.
    - Si un valor en 'paths' NO es string (por error), lo ignora.
    """
    raw = cfg.get("paths", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("La sección 'paths' debe ser un diccionario.")

    norm: Dict[str, str] = {}
    for key, val in raw.items():
        if not isinstance(val, str):
            # Solo normalizamos rutas que sean strings (evita ambigüedades)
            continue
        p = Path(val)
        if not p.is_absolute():
            p = (root / p).resolve()
        norm[key] = str(p)  # guardamos como string para facilitar serialización/prints
    return norm


def _ensure_base_dirs(paths: Dict[str, str]) -> None:
    """
    Crea directorios base si están definidos en 'paths'.
    - No crea archivos, SOLO carpetas típicas:
        'data_dir', 'datasets_dir', 'logs_dir', 'debug_dir'.
    - Es idempotente: si la carpeta ya existe, no pasa nada.
    """
    for key in ("data_dir", "datasets_dir", "logs_dir", "debug_dir"):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) API PÚBLICA (lo que usarás desde el resto del proyecto)
# ============================================================

# (Opcional) Cache simple para evitar releer disco muchas veces.
# Si no quieres cache, pon _CFG_CACHE = None siempre y retorna sin usarlo.
_CFG_CACHE: Dict[str, Any] | None = None

def load_config() -> Dict[str, Any]:
    """
    Carga la configuración del proyecto con este orden de prioridad:
      1) Data/config.yaml        (principal)
      2) Data/configInit.yaml    (fallback si no existe el principal)
      3) Data/config.local.yaml  (overlay opcional; no se versiona)

    Luego:
      - Normaliza todas las rutas en cfg['paths'] a absolutas.
      - Crea directorios base si están definidos (data_dir, datasets_dir, logs_dir, debug_dir).
      - Adjunta metadatos útiles en cfg["_meta"] (raíz del proyecto, archivo usado, override local).

    Retorna:
      - Un dict con toda la configuración del proyecto listo para usarse.
    """
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE

    root = _project_root()
    data_dir = root / "Data"

    primary   = data_dir / "config.yaml"        # Archivo principal recomendado
    fallback  = data_dir / "configInit.yaml"    # Alternativa si aún no renombraste
    localfile = data_dir / "config.local.yaml"  # Ajustes personales (no subir a git)

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

    # 2) Overlay local (opcional): agrega/reescribe claves sin tocar el archivo base
    if localfile.exists():
        _deep_update(cfg, _load_yaml(localfile))

    # 3) Normalizar rutas y crear directorios base
    paths_abs = _normalize_paths(cfg, root)
    cfg["paths"] = paths_abs
    _ensure_base_dirs(paths_abs)

    # 4) Metadatos: útiles para depurar qué archivo se usó y desde dónde se leyó
    cfg["_meta"] = {
        "project_root": str(root),
        "cfg_source": str(source),
        "cfg_local": str(localfile) if localfile.exists() else None,
    }

    _CFG_CACHE = cfg
    return cfg


def get_path(key: str) -> Path:
    """
    Atajo para obtener una ruta absoluta definida en cfg['paths'][key] como objeto Path.
    Ejemplo:
        get_path("data_dir") -> Path("/ruta/absoluta/al/proyecto/Data")
    Si la clave no existe, lanza KeyError con un mensaje claro.
    """
    cfg = load_config()
    try:
        return Path(cfg["paths"][key])
    except KeyError as e:
        raise KeyError(f"No existe 'paths.{key}' en la configuración.") from e


def project_root() -> Path:
    """Devuelve la raíz del proyecto como Path (atajo de _project_root())."""
    return _project_root()
