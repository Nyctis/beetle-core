"""The Mac-alone baseline: the same memory recipe on this machine with nothing else.

`beetle train` has no local path on a Mac, so this runs the project's own MLX
trainer (`sandbox/train_eval_mlx.py`) on the workload in workload.json. It
measures what a person gets by staying on one laptop.

    python experiments/evidence/mac_baseline.py --base qwen3.5-2b --out OUT.json

A watchdog holds the child to two limits and kills it the moment either breaks:

    --max-seconds   wall clock, model load included (default 900)
    --min-free      `memory_pressure` free percentage (default 8)
    --min-disk-gb   free disk, because a swapping Mac eats its own disk (default 4)
    --max-swap-mb   swap growth since the start, which stops a run before the machine
                    stalls: one sample came 8 s late at the worst of it (default 5000)

A run stopped by the watchdog is a result, not an error: it is written down with
the phase it died in, and the exit code is still 0.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("start", "load", "base_answers", "train", "adapter_answers", "done")


def free_percent() -> int | None:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
    found = re.search(r"free percentage: (\d+)%", out)
    return int(found[1]) if found else None


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else 0.0


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    found = re.search(r"used = ([\d.]+)M", out)
    return float(found[1]) if found else 0.0


def load_average() -> float:
    return os.getloadavg()[0]


def watch(a) -> int:
    status = Path(a.out).with_suffix(".status.json")
    status.unlink(missing_ok=True)
    samples_path = Path(a.out).with_suffix(".samples.jsonl")
    memory = Path(a.out).with_suffix(".mem.json")
    memory.unlink(missing_ok=True)
    cmd = [sys.executable, __file__, "--child", "--base", a.base, "--out", a.out, "--status", str(status),
           "--memory", str(memory)]
    if a.batch:
        cmd += ["--batch", str(a.batch)]
    t0 = time.time()
    swap0, free0, load0 = swap_used_mb(), free_percent(), [round(x, 2) for x in os.getloadavg()]
    child = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    stopped, low, peak_rss, peak_swap, loads, phase = None, 100 if free0 is None else free0, 0.0, swap0, [], "start"
    mlx_peak, step_at_end = 0.0, None
    with samples_path.open("w") as samples:
        while child.poll() is None:
            free, rss, swap = free_percent(), rss_mb(child.pid), swap_used_mb()
            elapsed = time.time() - t0
            try:
                phase = json.loads(status.read_text()).get("phase", phase)
            except (OSError, ValueError):
                pass
            try:
                inside = json.loads(memory.read_text())
                mlx_peak, step_at_end = max(mlx_peak, inside["peak_gb"]), inside["step"]
            except (OSError, ValueError):
                pass
            disk = shutil.disk_usage(Path(a.out).parent).free / 2**30
            low = min(low, 100 if free is None else free)
            peak_rss, peak_swap = max(peak_rss, rss), max(peak_swap, swap)
            loads.append(load_average())
            samples.write(json.dumps({"t": round(elapsed, 1), "free_pct": free, "rss_mb": round(rss, 1),
                                      "swap_mb": swap, "mlx_peak_gb": mlx_peak, "step": step_at_end, "disk_gb": round(disk, 1),
                                      "phase": phase, "load1": round(loads[-1], 2)}) + "\n")
            samples.flush()
            if free is not None and free < a.min_free:
                stopped = f"memory_pressure free {free}% is under {a.min_free}%"
            elif disk < a.min_disk_gb:
                stopped = f"disk free {disk:.1f} GiB is under {a.min_disk_gb} GiB"
            elif swap - swap0 > a.max_swap_mb:
                stopped = f"swap grew {swap - swap0:.0f} MB, over {a.max_swap_mb} MB"
            elif elapsed > a.max_seconds:
                stopped = f"wall clock over {a.max_seconds} s"
            if stopped:
                child.send_signal(signal.SIGKILL)
                child.wait()
                break
            time.sleep(a.every)
    report = json.loads(status.read_text()) if status.exists() else {}
    report.update(base=a.base, wall_s=round(time.time() - t0, 1), stopped_by_watchdog=stopped,
                  exit_code=child.returncode, started_at=int(t0), load_start=load0, free_pct_start=free0, free_pct_min=low,
                  rss_peak_mb=round(peak_rss, 1), swap_start_mb=swap0, swap_peak_mb=peak_swap,
                  mlx_peak_gb=mlx_peak, step_at_end=step_at_end, batch_override=a.batch,
                  load1_mean=round(sum(loads) / max(len(loads), 1), 2), load1_max=round(max(loads, default=0), 2),
                  phase_at_end=phase, limits={"max_seconds": a.max_seconds, "min_free_pct": a.min_free,
                                              "min_disk_gb": a.min_disk_gb, "max_swap_mb": a.max_swap_mb},
                  machine={"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                                 text=True).stdout.strip(),
                           "memory_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                                 text=True).stdout) / 2**30, 1),
                           "os": platform.mac_ver()[0]})
    report["completed"] = report.get("phase") == "done" and not stopped and child.returncode == 0
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report.get(k) for k in ("base", "completed", "wall_s", "phase_at_end", "stopped_by_watchdog",
                                                 "free_pct_min", "swap_start_mb", "swap_peak_mb", "mlx_peak_gb",
                                                 "step_at_end", "load_start", "base_score", "adapter", "training", "error")}))
    return 0


def child(a) -> int:
    sys.path[:0] = [str(ROOT / "sandbox"), str(ROOT / "common"), str(ROOT)]
    status_path = Path(a.status)
    state = {"phase": "start", "t": {}}
    t0 = time.time()
    steps = {"done": 0}

    def mark(phase: str, **extra) -> None:
        state["phase"] = phase
        state["t"][phase] = round(time.time() - t0, 1)
        state.update(extra)
        status_path.write_text(json.dumps(state, ensure_ascii=False))
        print(f"[{state['t'][phase]:7.1f}s] {phase}", flush=True)

    try:
        import mlx.core as mx
        import mlx_lm
        import model_profiles as mp
        import train_eval_mlx as te
        from mlx_lm import generate as mlx_generate
        from recipe_data import augment

        work = json.loads((Path(__file__).parent / "workload.json").read_text())
        profile = mp.get(a.base)
        memories = json.loads((ROOT / work["memories"]).read_text())
        hidden = json.loads((ROOT / work["hidden"]).read_text())
        recipe = profile.recipe()
        mark("load", mlx_repo=profile.mlx["repo"], mlx_lm=mlx_lm.__version__, mlx=mx.__version__)
        tok, model = te.load_base(profile.mlx["repo"])
        think = profile.chat.get("template", {})

        def report_memory() -> None:
            """Metal memory isn't in a process's RSS, and a killed child can't report at the end."""
            while True:
                Path(a.memory).write_text(json.dumps({"peak_gb": round(mx.get_peak_memory() / 2**30, 3),
                                                      "step": steps["done"]}))
                time.sleep(1)

        threading.Thread(target=report_memory, daemon=True).start()

        def answers(template: dict, max_tokens: int) -> list[str]:
            out = []
            for item in hidden:
                prompt = tok.apply_chat_template([{"role": "user", "content": item["question"]}], tokenize=False,
                                                 add_generation_prompt=True, **template)
                out.append(mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens).strip())
            return out

        def scored(texts: list[str]) -> dict:
            hits = [it["answer"].lower() in t.lower() for it, t in zip(hidden, texts)]
            return {"first6": sum(hits[:6]), "all": sum(hits), "n": len(hits)}

        mark("base_answers", peak_after_load_gb=round(mx.get_peak_memory() / 2**30, 3))
        before = answers(think, 32)
        mark("train", base_score=scored(before), base_sample=before[:2])
        pairs = augment(memories)
        if a.batch:
            recipe["optim"]["batch_size"] = a.batch
        counted = te.masked_loss

        def masked_loss(*args):
            steps["done"] += 1
            return counted(*args)

        te.masked_loss = masked_loss
        tuned_model, training = te.train_lora(model, tok, pairs, recipe, {"max_train_tokens": 10**9,
                                                                          "max_wall_seconds": 10**9}, print)
        loss = training.pop("loss_log")
        training.pop("checkpoint_root")
        mark("adapter_answers", training=training, loss_first=loss[0], loss_last=loss[-1],
             peak_after_train_gb=round(mx.get_peak_memory() / 2**30, 3), pairs=len(pairs),
             recipe={"lora": recipe["lora"], "optim": recipe["optim"], "max_seq_len": recipe["max_seq_len"],
                     "seed": recipe["seed"]})
        after = answers({}, 32)
        mark("done", adapter=scored(after), adapter_sample=after[:2],
             changed_first6=sum(b != c for b, c in zip(before[:6], after[:6])),
             peak_gb=round(mx.get_peak_memory() / 2**30, 3))
    except BaseException as e:  # a failure to load is the measurement here
        state["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        status_path.write_text(json.dumps(state, ensure_ascii=False))
        print(state["error"], file=sys.stderr, flush=True)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="a profile with an MLX build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=int, default=900)
    ap.add_argument("--min-free", type=int, default=8)
    ap.add_argument("--min-disk-gb", type=int, default=4)
    ap.add_argument("--max-swap-mb", type=int, default=5000)
    ap.add_argument("--every", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--batch", type=int, help="not the recipe: a smaller batch, to see what would fit")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--status", help=argparse.SUPPRESS)
    ap.add_argument("--memory", help=argparse.SUPPRESS)
    a = ap.parse_args()
    return child(a) if a.child else watch(a)


if __name__ == "__main__":
    sys.exit(main())
