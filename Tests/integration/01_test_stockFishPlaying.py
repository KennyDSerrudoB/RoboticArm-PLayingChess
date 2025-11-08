# Tests/integration/01_test_stockFishPlaying.py
"""
Modo interactivo: Juega Humano vs Stockfish desde consola.
- Puedes elegir el color (blancas o negras).
- Muestra el tablero orientado correctamente.
- Guarda la partida en Data/last_game.pgn al terminar.
"""

# --- Ajuste para permitir imports desde la raíz del proyecto ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------------

import time
import chess
import chess.engine
import chess.pgn
from Pc.common.config_loader import load_config


def draw_board(board: chess.Board, flip: bool = False):
    """Imprime el tablero desde la perspectiva del jugador."""
    if flip:
        flipped = board.mirror()
        print(flipped)
    else:
        print(board)
    print("\n")


def save_pgn(game: chess.pgn.Game, path: Path):
    """Guarda la partida en formato PGN."""
    with open(path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    print(f"💾 Partida guardada en: {path}")


def play_human_vs_stockfish():
    """Permite jugar humano vs Stockfish desde consola."""

    # === 1. Cargar configuración ===
    cfg = load_config()
    engine_path = Path(cfg["engine"]["bin"])
    if not engine_path.exists():
        print(f"❌ No se encontró el motor en {engine_path}")
        return

    # === 2. Inicializar motor ===
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    options = cfg["engine"].get("options", {})
    for k, v in options.items():
        try:
            engine.configure({k: v})
        except chess.engine.EngineError:
            pass  # Ignora opciones no compatibles

    # === 3. Límite de cálculo ===
    limits_cfg = cfg["engine"].get("limits", {})
    if limits_cfg.get("depth"):
        limit = chess.engine.Limit(depth=int(limits_cfg["depth"]))
    else:
        limit = chess.engine.Limit(time=float(limits_cfg.get("time_ms", 200)) / 1000.0)

    # === 4. Elegir color ===
    color_input = input("¿Quieres jugar con las blancas o las negras? (b/n): ").strip().lower()
    human_is_white = color_input in ("b", "blancas", "white", "w")
    board = chess.Board()
    flip_view = not human_is_white

    print("\n♟️  PARTIDA: HUMANO vs STOCKFISH ♟️")
    print("Tú juegas con:", "BLANCAS" if human_is_white else "NEGRAS")
    draw_board(board, flip=flip_view)

    # === 5. Crear estructura PGN ===
    game = chess.pgn.Game()
    game.headers["Event"] = "Human vs Stockfish"
    game.headers["White"] = "Humano" if human_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if human_is_white else "Humano"
    node = game

    # === 6. Si el humano es negro, Stockfish abre ===
    if not human_is_white:
        print("🤖 Stockfish abre la partida...")
        result = engine.play(board, limit)
        move = result.move
        move_san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        print("Stockfish juega:", move_san)
        draw_board(board, flip=flip_view)

    # === 7. Bucle principal ===
    while not board.is_game_over(claim_draw=True):
        if (board.turn == chess.WHITE and human_is_white) or (board.turn == chess.BLACK and not human_is_white):
            # --- Turno humano ---
            print("Tu turno (formato UCI, ej. e2e4, o 'salir'):")
            user_move = input("→ ").strip().lower()

            if user_move == "salir":
                print("🚪 Partida terminada por el usuario.")
                break

            try:
                move = chess.Move.from_uci(user_move)
                if move not in board.legal_moves:
                    print("❌ Jugada ilegal o fuera de turno. Intenta otra.")
                    continue

                move_san = board.san(move)  # ← CORREGIDO: calcular SAN antes del push
                board.push(move)
                node = node.add_variation(move)
                print("\nTu jugada:", move_san)
                draw_board(board, flip=flip_view)
            except ValueError:
                print("❌ Formato inválido. Usa algo como e2e4 o g1f3.")
                continue
        else:
            # --- Turno del motor ---
            print("🤖 Stockfish está pensando...")
            result = engine.play(board, limit)
            move = result.move
            move_san = board.san(move)
            board.push(move)
            node = node.add_variation(move)
            print("\nStockfish juega:", move_san)
            draw_board(board, flip=flip_view)
            time.sleep(0.3)

    # === 8. Fin de partida ===
    if board.is_game_over():
        print("\n=== PARTIDA FINALIZADA ===")
        print("Resultado:", board.result())
        print("Motivo:", board.outcome().termination)
        game.headers["Result"] = board.result()

    # Guardar la partida
    save_pgn(game, Path("Data/last_game.pgn"))
    engine.quit()


if __name__ == "__main__":
    play_human_vs_stockfish()
