"""Start the app:  python run.py   then open http://localhost:8000"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-restart when code changes (development)")
    args = ap.parse_args()
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=args.reload)
