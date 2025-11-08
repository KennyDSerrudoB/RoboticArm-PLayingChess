# Tests/unit/test_stockfish_full.py
from pathlib import Path
import sys
import platform
import subprocess
import pytest
import chess
import chess.engine
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ---------------------------------------------------
# Usamos tu loader central
from Pc.common.config_loader import load_config

def _cfg():
    return load_config()

def _engine_path() -> Path:
    cfg = _cfg()
    path_str = (
        cfg.get("engine", {}).get("bin")
        or cfg.get("paths", {}).get("stockfish_bin", "")
    )
    return Path(path_str).resolve()

def _uci_options():
    """Lee opciones UCI del YAML y normaliza tipos (int/bool)."""
    cfg = _cfg()
    opts = (cfg.get("engine", {}).get("options") or {}).copy()
    for k, v in list(opts.items()):
        if isinstance(v, str) and v.isdigit():
            opts[k] = int(v)
        if isinstance(v, str) and v.lower() in ("true", "false"):
            opts[k] = (v.lower() == "true")
    return opts

def _limits():
    """Construye chess.engine.Limit a partir de engine.limits."""
    cfg = _cfg()
    lims = cfg.get("engine", {}).get("limits", {}) or {}
    if lims.get("time_ms") is not None:
        t = max(0.01, float(lims["time_ms"]) / 1000.0)
        return chess.engine.Limit(time=t)
    if lims.get("depth") is not None:
        return chess.engine.Limit(depth=int(lims["depth"]))
    if lims.get("nodes") is not None:
        return chess.engine.Limit(nodes=int(lims["nodes"]))
    # default si nada está definido
    return chess.engine.Limit(time=0.2)

def _open_engine():
    """Abre el motor y aplica opciones UCI. Hace skip si no hay binario."""
    path = _engine_path()
    if not path.exists():
        pytest.skip("No se encontró engine.bin en Data/config.yaml (o la ruta no existe).")
    if platform.system() == "Windows" and path.suffix.lower() != ".exe":
        pytest.skip("En Windows se espera .exe para Stockfish.")
    engine = chess.engine.SimpleEngine.popen_uci(str(path))
    try:
        engine.configure(_uci_options())
    except chess.engine.EngineError:
        # Algunas builds no soportan todas las opciones; las ignoramos.
        pass
    return engine

# =============== TESTS ===============

def test_engine_binary_exists():
    """Existe el binario y es archivo."""
    p = _engine_path()
    if not p.exists():
        pytest.skip("Stockfish no encontrado. Ajusta engine.bin en Data/config.yaml.")
    assert p.is_file()

def test_engine_cli_version():
    """El binario responde por CLI (útil para diagnosticar instalación/permisos)."""
    p = _engine_path()
    if not p.exists():
        pytest.skip("Stockfish no encontrado.")
    try:
        out = subprocess.check_output([str(p), "--help"], stderr=subprocess.STDOUT, timeout=5)
    except subprocess.CalledProcessError as e:
        out = e.output
    except Exception:
        pytest.skip("La build no soporta '--help'; continuamos con UCI.")
        return
    assert b"Stockfish" in out or b"uci" in out or b"help" in out

def test_engine_basic_move_from_startpos():
    """Desde la posición inicial, el motor devuelve una jugada legal."""
    engine = _open_engine()
    try:
        board = chess.Board()
        limit = _limits()
        result = engine.play(board, limit)
        assert result.move is not None, "No devolvió jugada."
        assert result.move in board.legal_moves, "Jugada ilegal."
    finally:
        engine.quit()

def test_engine_mate_in_one():
    """Reconoce un mate en 1 en una posición simple."""
    engine = _open_engine()
    try:
        # FEN con mate inmediato para blancas: Dh7#
        fen = "r6k/7P/8/7Q/8/8/8/R3K3 w Q - 0 1"
        board = chess.Board(fen=fen)
        result = engine.play(board, _limits())
        assert result.move is not None, "No devolvió jugada en táctica simple."
        san = board.san(result.move)
        assert san.endswith("#"), f"Se esperaba mate en 1; el motor jugó {san!r}"
    finally:
        engine.quit()

def test_engine_determinism_with_depth_if_set():
    """
    Si en YAML usas limits.depth (en lugar de time_ms), la primera jugada
    debería repetirse (más determinista). Si no está configurado, se hace skip.
    """
    cfg = _cfg()
    depth = (cfg.get("engine", {}).get("limits", {}) or {}).get("depth")
    if depth is None:
        pytest.skip("Este test requiere limits.depth en Data/config.yaml (no time_ms).")

    engine = _open_engine()
    try:
        board1 = chess.Board()
        board2 = chess.Board()
        limit = chess.engine.Limit(depth=int(depth))
        res1 = engine.play(board1, limit)
        res2 = engine.play(board2, limit)
        assert res1.move == res2.move, f"No es determinista: {res1.move} vs {res2.move}"
    finally:
        engine.quit()
