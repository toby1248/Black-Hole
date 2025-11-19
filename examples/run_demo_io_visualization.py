#!/usr/bin/env python3
"""
Small wrapper to run examples/demo_io_visualization.py and keep the CLI open so you can
read debug output. This avoids changing the original demo file.

Usage:
  python examples/run_demo_io_visualization.py            # run and don't pause at the end
  python examples/run_demo_io_visualization.py --pause   # wait for Enter before exiting
  python examples/run_demo_io_visualization.py --log out.log  # save a copy of stdout/stderr to out.log
  python examples/run_demo_io_visualization.py --debug  # set an env var that demos may read for extra debug

Notes:
- The wrapper streams live output from the demo and optionally writes it to a logfile.
- Works on Linux/macOS/Windows (requires Python 3.6+).
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

def stream_process(cmd, logfile_path=None, env=None):
    # Start subprocess and stream its stdout/stderr line-by-line, also optionally logging to a file.
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    log_fh = None
    if logfile_path:
        log_fh = open(logfile_path, "w", encoding="utf-8")

    try:
        for line in popen.stdout:
            # print live to caller's stdout (no extra buffering)
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
        popen.wait()
        return popen.returncode
    finally:
        if log_fh:
            log_fh.close()

def main():
    parser = argparse.ArgumentParser(description="Run demo_io_visualization with a pause/logger wrapper.")
    parser.add_argument("--pause", action="store_true",
                        help="Wait for Enter before exiting so you can read debug output.")
    parser.add_argument("--log", "-l", metavar="FILE", help="Save a copy of stdout/stderr to FILE.")
    parser.add_argument("--debug", action="store_true", help="Set DEBUG=1 in the child environment.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for the demo script.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to the demo script.")
    args = parser.parse_args()

    demo_path = Path(__file__).resolve().parent / "demo_io_visualization.py"
    if not demo_path.exists():
        print(f"ERROR: demo script not found at: {demo_path}", file=sys.stderr)
        sys.exit(2)

    cmd = [args.python, str(demo_path)] + args.extra

    # Prepare environment: optionally set DEBUG=1 for child
    env = os.environ.copy()
    if args.debug:
        env["DEBUG"] = "1"

    print("Running:", " ".join(cmd))
    if args.log:
        print("Logging to:", args.log)
    print("---- demo output ----")

    rc = stream_process(cmd, logfile_path=args.log, env=env)

    print("---- demo finished (rc={}) ----".format(rc))
    # Only pause if requested and we have an interactive terminal
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False

    if args.pause and is_tty:
        try:
            input("Press Enter to exit...")
        except EOFError:
            # non-interactive environment
            pass

    sys.exit(rc)

if __name__ == "__main__":
    main()