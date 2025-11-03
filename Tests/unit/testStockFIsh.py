import shutil
import chess

import chess.engine

def test_stockfish_basic():
    """Verifica que el motor Stockfish responde y puede calcular una jugada"""
    engine_path = shutil.which  ("stockfish") or "/usr/games/stockfish"
    print(f"Motor detectado: {engine_path}")

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 4, "Hash": 256, "Skill Level": 15})

    board = chess.Board()
    print("Tablero inicial:")
    print(board)

    # Pide una jugada con tiempo limitado
    result = engine.play(board, chess.engine.Limit(time=0.5))
    move = result.move.uci()
    board.push(result.move)

    print("Jugada sugerida:", move)
    print("Nuevo tablero:")
    print(board)

    engine.quit()

    # Aserción mínima para pruebas automatizadas
    assert len(move) in (4, 5), "Movimiento UCI inválido"
