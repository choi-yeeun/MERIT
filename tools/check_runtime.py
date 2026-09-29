#!/usr/bin/env python3
"""Report what this machine or container can actually do: GPUs, deps, network."""

import argparse
import os
import shutil
import socket
import ssl
import sys

ENDPOINTS = [
    ("api.openai.com", "gpt-* models"),
    ("huggingface.co", "model weights and the EgoLife dataset"),
    ("generativelanguage.googleapis.com", "gemini-* models (optional)"),
]


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status:<4}] {label}{('  ' + detail) if detail else ''}")


def reachable(host: str, port: int = 443, timeout: float = 6.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(sock, server_hostname=host):
                return True, "TLS handshake ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-network", action="store_true")
    args = p.parse_args()

    problems = []

    print("--- python / packages ---")
    line("ok", "python", sys.version.split()[0])
    for mod in ("torch", "torchvision", "transformers", "sentence_transformers",
                "openai", "decord", "flash_attn", "qwen_vl_utils"):
        try:
            m = __import__(mod)
            line("ok", mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            line("FAIL", mod, repr(e))
            problems.append(f"{mod} does not import")
    try:
        import merit
        line("ok", "merit", merit.__version__)
    except Exception as e:  # noqa: BLE001
        line("FAIL", "merit", repr(e))
        problems.append("merit does not import")

    print("\n--- gpu ---")
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            line("ok", "cuda", f"{torch.version.cuda}, {n} device(s)")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                line("ok", f"  gpu {i}",
                     f"{props.name}, {props.total_memory / 1e9:.0f} GB, sm_{props.major}{props.minor}")
        else:
            line("FAIL", "cuda", "torch.cuda.is_available() is False")
            problems.append("no visible GPU (pass --gpus all to docker run)")
    except Exception as e:  # noqa: BLE001
        line("FAIL", "cuda", repr(e))
        problems.append("cuda probe failed")

    print("\n--- tools ---")
    for exe, why in (("ffmpeg", "Video-MME / LVBench preprocessing"),
                     ("hf", "dataset and model download")):
        path = shutil.which(exe)
        line("ok" if path else "warn", exe, path or f"missing — needed for {why}")

    print("\n--- api keys ---")
    for var, why in (("OPENAI_API_KEY", "gpt-* models"),
                     ("GOOGLE_API_KEY", "gemini-* models (optional)")):
        present = bool(os.environ.get(var))
        status = "ok" if present else ("warn" if "optional" in why else "FAIL")
        line(status, var, "set" if present else f"not set — needed for {why}")
        if status == "FAIL":
            problems.append(f"{var} is not set (docker run -e {var})")

    if not args.skip_network:
        print("\n--- network ---")
        for host, why in ENDPOINTS:
            ok, detail = reachable(host)
            optional = "optional" in why
            line("ok" if ok else ("warn" if optional else "FAIL"),
                 host, detail if ok else f"{detail}  (needed for {why})")
            if not ok and not optional:
                problems.append(f"cannot reach {host} — needed for {why}")

    print("\n--- disk ---")
    for path in ("/workspace", os.getcwd()):
        if os.path.isdir(path):
            total, used, free = shutil.disk_usage(path)
            line("ok", path, f"{free / 1e9:.0f} GB free of {total / 1e9:.0f} GB")

    print()
    if problems:
        print("Blocking issues:")
        for x in dict.fromkeys(problems):
            print(f"  - {x}")
        return 1
    print("Runtime looks usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
