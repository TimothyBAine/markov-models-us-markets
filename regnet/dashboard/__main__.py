import argparse

from .app import create_app


def main():
    p = argparse.ArgumentParser(prog="python -m regnet.dashboard")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    app = create_app()
    print(f"Regime network dashboard on http://127.0.0.1:{a.port}")
    app.run(host="127.0.0.1", port=a.port, debug=a.debug)


if __name__ == "__main__":
    main()
