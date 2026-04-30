import argparse
import os

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="NeSyPaperGraph desktop sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
