#!/usr/bin/env python3
"""Live, dependency-free local dashboard for steel-ball training runs."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "training" / "runs" / "detect"
RUN_LABELS = {
    "steel_ball_reference_yolo26n_1024_live": "YOLO26 1024",
    "steel_ball_reference_yolo26n_1024_failed_missing_polars": "YOLO26 首次启动",
    "steel_ball_reference_yolo26n_1024": "YOLO26 初始运行",
    "steel_ball_reference_yolo11n_640_fast": "YOLO11 快速 640",
    "steel_ball_reference_yolo11n_1024": "YOLO11 高清 1024",
}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GPU_CACHE = {"at": 0.0, "data": None}


def parse_gpu_output(output: str) -> dict:
    """Parse nvidia-smi CSV output and select the busiest visible GPU."""
    snapshots = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) < 7:
            continue
        try:
            index = int(row[0].strip())
            utilization = float(row[2].strip())
            memory_used = float(row[3].strip())
            memory_total = float(row[4].strip())
            temperature = float(row[5].strip())
        except ValueError:
            continue
        try:
            fan_percent = float(row[6].strip())
        except ValueError:
            fan_percent = 0.0
        snapshots.append(
            {
                "available": True,
                "index": index,
                "name": row[1].strip(),
                "utilization": max(0.0, min(100.0, utilization)),
                "memory_used_mb": max(0.0, memory_used),
                "memory_total_mb": max(0.0, memory_total),
                "memory_percent": (
                    max(0.0, min(100.0, memory_used / memory_total * 100.0))
                    if memory_total
                    else 0.0
                ),
                "temperature": temperature,
                "fan_percent": max(0.0, min(100.0, fan_percent)),
            }
        )
    if not snapshots:
        return {"available": False}
    return max(snapshots, key=lambda item: (item["memory_used_mb"], item["utilization"]))


def read_gpu(cache_seconds: float = 2.0) -> dict:
    """Read real NVIDIA telemetry without making every API request spawn a process."""
    now = time.monotonic()
    cached = GPU_CACHE["data"]
    if cached is not None and now - GPU_CACHE["at"] < cache_seconds:
        return cached
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        data = parse_gpu_output(result.stdout) if result.returncode == 0 else {"available": False}
    except (OSError, subprocess.SubprocessError):
        data = {"available": False}
    GPU_CACHE.update(at=now, data=data)
    return data


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0").strip())
    except (AttributeError, ValueError):
        return 0.0


def tail(path: Path, limit: int = 750_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def clean_log(logs: str) -> str:
    return ANSI_RE.sub("", logs).replace("\r", "\n")


def terminal_view(logs: str, synthetic: str, limit: int = 80) -> str:
    """Turn Rich/TQDM redraws into a stable terminal history."""
    if not logs:
        return synthetic
    text = clean_log(logs)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique: list[str] = []
    for line in lines:
        if not unique or unique[-1] != line:
            unique.append(line)
    return "\n".join(unique[-limit:]) or synthetic


def parse_progress(logs: str) -> dict:
    """Extract the most recent Ultralytics epoch/batch progress line."""
    for line in reversed(clean_log(logs).splitlines()):
        fractions = list(re.finditer(r"(?<![\d.])(\d+)/(\d+)(?![\d.])", line))
        if len(fractions) < 2:
            continue
        epoch_match, batch_match = fractions[0], fractions[-1]
        losses = re.search(
            r"\d+/\d+\s+([\d.]+G)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
            line,
        )
        rate = re.search(r"([\d.]+)it/s", line)
        eta = re.search(r"<([0-9:]+)", line)
        return {
            "epoch_current": int(epoch_match.group(1)),
            "epoch_total": int(epoch_match.group(2)),
            "batch_current": int(batch_match.group(1)),
            "batch_total": int(batch_match.group(2)),
            "gpu_memory": losses.group(1) if losses else "—",
            "box_loss": float(losses.group(2)) if losses else 0.0,
            "cls_loss": float(losses.group(3)) if losses else 0.0,
            "aux_loss": float(losses.group(4)) if losses else 0.0,
            "batch_rate": float(rate.group(1)) if rate else 0.0,
            "eta": eta.group(1) if eta else "—",
        }
    return {}


def classify_status(logs: str, age_seconds: float, has_results: bool) -> str:
    """Classify a run from terminal markers and its log heartbeat."""
    lowered = logs.lower()
    if "goal_early_stop triggered=plateau_after_goal" in lowered or "epochs completed in" in lowered:
        return "completed"
    if age_seconds <= 15:
        return "running" if logs.strip() else "starting"
    if "traceback" in lowered or "runtimeerror:" in lowered or "modulenotfounderror:" in lowered:
        return "failed"
    if has_results:
        return "completed"
    return "stopped"


def run_logs(directory: Path) -> tuple[str, float, float | None]:
    paths = [directory / "train.out.log", directory / "train.err.log"]
    logs = "\n".join(tail(path) for path in paths if path.is_file())
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    updated = max(mtimes) if mtimes else None
    age = max(0.0, time.time() - updated) if updated else 999_999.0
    return logs, age, updated


def read_total_epochs(directory: Path, fallback: int = 60) -> int:
    args_file = directory / "args.yaml"
    if not args_file.is_file():
        return fallback
    match = re.search(r"(?m)^epochs:\s*(\d+)\s*$", args_file.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else fallback


def read_rows(results: Path) -> list[dict[str, str]]:
    if not results.is_file():
        return []
    with results.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_run(run_id: str, name: str, directory: Path) -> dict:
    results = directory / "results.csv"
    rows = read_rows(results)
    raw_epochs = [int(number(row, "epoch")) for row in rows]
    offset = 1 if raw_epochs and raw_epochs[0] == 0 else 0
    epochs = [epoch + offset for epoch in raw_epochs]
    series = {
        "epochs": epochs,
        "map50": [number(row, "metrics/mAP50(B)") for row in rows],
        "precision": [number(row, "metrics/precision(B)") for row in rows],
        "recall": [number(row, "metrics/recall(B)") for row in rows],
    }
    logs, age, updated_timestamp = run_logs(directory)
    progress = parse_progress(logs)
    total_epochs = progress.get("epoch_total") or read_total_epochs(directory)
    current_epoch = progress.get("epoch_current") or (epochs[-1] if epochs else 0)
    status = classify_status(logs, age, bool(rows))
    last = rows[-1] if rows else {}
    plateau_matches = re.findall(
        r"GOAL_EARLY_STOP epoch=(\d+) goal=(yes|no) stalled=(\d+)/(\d+)", clean_log(logs)
    )
    plateau = plateau_matches[-1] if plateau_matches else None
    metrics = {
        "map50": number(last, "metrics/mAP50(B)"),
        "precision": number(last, "metrics/precision(B)"),
        "recall": number(last, "metrics/recall(B)"),
        "map5095": number(last, "metrics/mAP50-95(B)"),
    }
    synthetic = "[monitor] waiting for training output..."
    if rows:
        synthetic = "[monitor] epoch=%d mAP50=%.4f precision=%.4f recall=%.4f" % (
            epochs[-1], metrics["map50"], metrics["precision"], metrics["recall"]
        )
    return {
        "id": run_id,
        "name": name,
        "status": status,
        "updated_at": datetime.fromtimestamp(updated_timestamp).strftime("%H:%M:%S") if updated_timestamp else "—",
        "_sort_time": updated_timestamp or directory.stat().st_mtime,
        "log_age_seconds": round(age, 2),
        "epoch": {"current": current_epoch, "total": total_epochs},
        "batch": {
            "current": progress.get("batch_current", 0),
            "total": progress.get("batch_total", 0),
            "rate": progress.get("batch_rate", 0.0),
            "eta": progress.get("eta", "—"),
        },
        "runtime": {
            "gpu_memory": progress.get("gpu_memory", "—"),
            "box_loss": progress.get("box_loss", 0.0),
            "cls_loss": progress.get("cls_loss", 0.0),
        },
        "metrics": metrics,
        "plateau": {
            "goal_reached": bool(plateau and plateau[1] == "yes"),
            "stalled": int(plateau[2]) if plateau else 0,
            "patience": int(plateau[3]) if plateau else 7,
        },
        "series": series,
        "terminal": terminal_view(logs, synthetic),
    }


def collect_runs() -> dict:
    runs = []
    for run_id, name in RUN_LABELS.items():
        directory = RUNS_ROOT / run_id
        if directory.is_dir():
            runs.append(read_run(run_id, name, directory))
    active = [run for run in runs if run["status"] in {"running", "starting"}]
    history = [run for run in runs if run["status"] not in {"running", "starting"}]
    active.sort(key=lambda run: run["_sort_time"], reverse=True)
    history.sort(key=lambda run: run["_sort_time"], reverse=True)
    for run in active + history:
        run.pop("_sort_time", None)
    return {
        "generated_at": datetime.now().strftime("%H:%M:%S"),
        "gpu": read_gpu(),
        "active": active,
        "history": history,
    }


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>钢珠训练台</title><style>
:root{--sun:#efd48e;--paper:#fff4d7;--paper2:#f8e7bd;--ink:#392b24;--muted:#786052;--rule:#b9906b;--orange:#d87352;--deep:#9c432d;--green:#56866f;--gold:#bd8b2e;--screen:#26211e;--screen2:#302822;--screenInk:#f2d2a8;--error:#b64235}*{box-sizing:border-box}html{min-width:320px}body{margin:0;color:var(--ink);background-color:var(--sun);background-image:linear-gradient(rgba(77,56,45,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(77,56,45,.045) 1px,transparent 1px);background-size:32px 32px;font-family:"Microsoft YaHei",ui-sans-serif,system-ui,sans-serif;letter-spacing:0}.top{position:relative;border-bottom:2px solid var(--ink);background:rgba(239,212,142,.96)}.top-inner{max-width:1440px;margin:auto;padding:24px 30px 18px;min-height:126px}.brand small{font:700 10px ui-monospace,Consolas,monospace;letter-spacing:.13em;color:var(--deep)}h1{margin:5px 0 0;font:600 34px/1.05 Georgia,"Songti SC",serif}.nav-row{display:flex;align-items:center;gap:10px;margin-top:17px;padding-right:185px;min-width:0}.segments,.models{display:flex;gap:5px;min-width:0;overflow-x:auto;scrollbar-width:none}.segments::-webkit-scrollbar,.models::-webkit-scrollbar{display:none}button{font:600 11px "Microsoft YaHei",sans-serif;color:var(--ink);background:var(--paper);border:1px solid var(--rule);min-height:34px;padding:7px 12px;cursor:pointer;border-radius:3px;white-space:nowrap}button:hover{border-color:var(--deep)}button:focus-visible{outline:3px solid rgba(216,115,82,.35);outline-offset:2px}.segment.active,.model-button.active{color:#fffaf0;background:var(--orange);border-color:var(--deep)}.divider{width:1px;height:28px;background:var(--rule);flex:0 0 auto}.cat-zone{position:absolute;right:max(30px,calc((100vw - 1380px)/2));top:14px;width:154px;height:57px;border-bottom:1px dashed rgba(120,80,57,.55)}.cat{position:absolute;bottom:-2px;left:0;font-size:29px;line-height:1;filter:drop-shadow(0 4px 2px rgba(75,52,41,.18));transform:scaleX(-1);animation:catRun 3.6s ease-in-out infinite}.cat-zone.running:after{content:"● LIVE";position:absolute;right:0;top:49px;font:700 9px ui-monospace,Consolas,monospace;color:var(--green)}.cat-zone.completed .cat,.cat-zone.failed .cat,.cat-zone.stopped .cat{animation:none;left:108px;transform:none}.cat-zone.completed:after{content:"已完成";position:absolute;right:0;top:49px;font:9px ui-monospace,monospace;color:var(--green)}.cat-zone.failed:after{content:"训练异常";position:absolute;right:0;top:49px;font:9px ui-monospace,monospace;color:var(--error)}@keyframes catRun{0%{left:0;transform:scaleX(-1) translateY(0)}43%{left:116px;transform:scaleX(-1) translateY(-3px)}50%{left:116px;transform:scaleX(1)}93%{left:0;transform:scaleX(1) translateY(-3px)}100%{left:0;transform:scaleX(-1)}}main{max-width:1440px;margin:auto;padding:22px 30px 38px}.empty{min-height:50vh;display:grid;place-items:center;text-align:center;color:var(--muted)}.empty strong{display:block;font:24px Georgia,serif;color:var(--ink);margin-bottom:7px}.run-title{display:flex;justify-content:space-between;align-items:end;gap:20px;padding-bottom:15px;border-bottom:1px solid var(--rule)}.run-title h2{margin:0;font:600 25px Georgia,"Songti SC",serif}.run-title p{margin:5px 0 0;color:var(--muted);font-size:12px}.status{font:700 10px ui-monospace,Consolas,monospace;color:var(--green)}.status.failed{color:var(--error)}.progress-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;padding:17px 0 0}.progress-panel{background:var(--paper);border:1px solid var(--rule);padding:15px}.progress-head{display:flex;justify-content:space-between;align-items:end}.progress-head strong{font:600 27px Georgia,serif;color:var(--deep)}.progress-head span{font:600 21px Georgia,serif;color:var(--deep)}.progress-head small{display:block;color:var(--muted);font:9px ui-monospace,Consolas,monospace;margin-top:3px}.track{height:14px;margin-top:11px;background:#d6b77f;border:1px solid #bf976a;overflow:hidden;position:relative}.fill{height:100%;width:0;background:var(--orange);position:relative;transition:width .2s linear;box-shadow:5px 0 17px rgba(216,115,82,.48)}.fill:before{content:"";position:absolute;inset:0;background:repeating-linear-gradient(120deg,transparent 0 11px,rgba(255,248,222,.45) 12px 16px,transparent 17px 29px);animation:flow .75s linear infinite}.fill:after{content:"";position:absolute;right:-4px;top:1px;width:9px;height:9px;border-radius:50%;background:#fff7d9;box-shadow:0 0 0 4px rgba(216,115,82,.38),0 0 15px #c65c3d;animation:pulse .85s ease-in-out infinite}.track.paused .fill:before,.track.paused .fill:after{animation-play-state:paused}@keyframes flow{to{transform:translateX(29px)}}@keyframes pulse{50%{transform:scale(.68);opacity:.65}}.estimate{margin-top:7px;color:var(--muted);font:8px ui-monospace,Consolas,monospace}.metrics{display:grid;grid-template-columns:repeat(4,1fr);margin-top:14px;border:1px solid var(--rule);background:var(--paper)}.metric{padding:14px 16px}.metric+.metric{border-left:1px solid var(--rule)}.metric b{display:block;font:600 25px Georgia,serif;color:var(--deep);transition:opacity .2s}.metric small{font:9px ui-monospace,Consolas,monospace;color:var(--muted)}.metric.changed b{animation:numberPulse .42s ease-out}@keyframes numberPulse{50%{opacity:.45;transform:translateY(-2px)}}.runtime{display:flex;gap:18px;flex-wrap:wrap;padding:11px 0;color:var(--muted);font:10px ui-monospace,Consolas,monospace}.runtime b{color:var(--ink);font-weight:600}.terminal-head,.chart-head{display:flex;align-items:end;justify-content:space-between;gap:16px}.terminal-head{padding:8px 0}.terminal-head span,.chart-head span{font:9px ui-monospace,Consolas,monospace;color:var(--muted)}.terminal{height:260px;overflow:auto;margin:0;padding:14px 16px;background-color:var(--screen);background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px);background-size:100% 22px;color:var(--screenInk);border-left:4px solid var(--orange);font:10px/1.62 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;tab-size:2}.terminal.error{border-left-color:var(--error);color:#ffd0c3}.terminal::-webkit-scrollbar{width:9px}.terminal::-webkit-scrollbar-thumb{background:#6d5747}.chart-section{margin-top:24px;padding-top:17px;border-top:1px solid var(--rule)}.chart-head h3{margin:0;font:600 20px Georgia,serif}.legend{display:flex;gap:14px;flex-wrap:wrap}.legend i{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:5px}.chart-box{position:relative;height:310px;margin-top:11px;background:var(--paper);border:1px solid var(--rule)}canvas{display:block;width:100%;height:100%}.tooltip{position:absolute;display:none;z-index:3;padding:8px 10px;background:var(--screen);color:var(--screenInk);border-left:3px solid var(--orange);font:9px/1.55 ui-monospace,Consolas,monospace;pointer-events:none}.chart-empty{position:absolute;inset:0;display:none;place-items:center;color:var(--muted);font-size:12px}.heartbeat{display:inline-flex;align-items:center;gap:6px}.heartbeat i{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(86,134,111,.18);animation:beat 1.5s steps(1) infinite}@keyframes beat{50%{opacity:.35}}@media(max-width:800px){.top-inner{padding:19px 16px 15px;min-height:142px}.nav-row{padding-right:0;margin-top:48px}.cat-zone{right:16px;top:8px;width:105px}.cat{animation:none!important;left:67px!important;transform:none!important}main{padding:16px}.progress-grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.metric:nth-child(3){border-left:0;border-top:1px solid var(--rule)}.metric:nth-child(4){border-top:1px solid var(--rule)}.run-title{align-items:start;flex-direction:column}.chart-box{height:245px}.terminal{height:225px}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}.cat{left:108px!important;transform:none!important}}
html,body{max-width:100%;overflow-x:hidden}.top,.top-inner,main{width:100%;max-width:100vw}main>section,.run-title,.run-title>div,.progress-grid,.progress-panel,.progress-head,.terminal-head,.chart-head,.chart-section,.chart-box{min-width:0;max-width:100%}.run-title p{max-width:100%;overflow-wrap:anywhere;word-break:break-all}
.console-grid{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(320px,.72fr);gap:14px;align-items:stretch}.console-column{min-width:0}.gpu-panel{--fan-speed:1.25s;min-width:0;min-height:294px;padding:13px 14px 16px;color:var(--screenInk);background:var(--screen);border:1px solid #6e5949;border-left:4px solid var(--green);overflow:hidden;position:relative}.gpu-panel.unavailable{border-left-color:var(--muted)}.gpu-panel-head{display:flex;align-items:start;justify-content:space-between;gap:12px;margin-bottom:11px}.gpu-panel-head small{display:block;color:#b89a7d;font:8px ui-monospace,Consolas,monospace;letter-spacing:.08em}.gpu-panel-head strong{display:block;max-width:230px;margin-top:3px;color:#ffe0b5;font:600 12px/1.25 ui-monospace,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.gpu-live{color:#7fb094;font:700 8px ui-monospace,Consolas,monospace;white-space:nowrap}.gpu-readouts{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}.gpu-stat{min-width:0}.gpu-stat-head{display:flex;justify-content:space-between;gap:8px;color:#bea287;font:8px ui-monospace,Consolas,monospace}.gpu-stat-head b{color:#ffe0b5;font-size:11px}.gpu-meter{height:5px;margin-top:5px;background:#53463d;overflow:hidden}.gpu-meter i{display:block;height:100%;width:0;background:var(--orange);transition:width .5s ease}.gpu-stat:nth-child(2) .gpu-meter i{background:var(--green)}.gpu-shell{height:154px;position:relative;display:flex;align-items:center;justify-content:center;gap:clamp(14px,2.2vw,28px);padding:16px 24px 20px;border:1px solid #7b6452;background-color:#332b26;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:12px 12px;box-shadow:inset 0 0 0 4px #29221e}.gpu-shell:before{content:"RTX / DUAL FAN";position:absolute;top:7px;left:10px;color:#9e8169;font:7px ui-monospace,Consolas,monospace;letter-spacing:.12em}.gpu-shell:after{content:"";position:absolute;bottom:-1px;left:27%;width:46%;height:6px;background:repeating-linear-gradient(90deg,#bd8b2e 0 6px,#6c5424 6px 8px)}.gpu-fan{width:clamp(78px,7.4vw,104px);aspect-ratio:1;border:1px solid #826b59;border-radius:50%;position:relative;background:#211c19;box-shadow:0 0 0 5px #2a231f,inset 0 0 15px #110f0d}.gpu-fan-rotor{position:absolute;inset:10px;border-radius:50%;background:repeating-conic-gradient(from 8deg,#c66d4e 0 8deg,#594036 9deg 25deg,transparent 26deg 44deg);animation:gpuSpin var(--fan-speed) linear infinite;animation-play-state:paused}.gpu-fan-rotor:after{content:"";position:absolute;inset:35%;border-radius:50%;background:#d6b77f;border:3px solid #463830;box-shadow:0 0 0 3px #a15842}.gpu-panel.running .gpu-fan-rotor{animation-play-state:running}.gpu-foot{display:flex;justify-content:space-between;gap:8px;margin-top:7px;color:#b89a7d;font:8px ui-monospace,Consolas,monospace}.gpu-foot b{color:#ffe0b5}@keyframes gpuSpin{to{transform:rotate(1turn)}}@media(max-width:1050px){.console-grid{grid-template-columns:minmax(0,1fr)}.gpu-panel{min-height:0}.gpu-shell{height:170px}.gpu-fan{width:clamp(82px,18vw,116px)}}@media(max-width:560px){.progress-head strong{font-size:23px}.progress-head span{font-size:18px}.metric{padding:12px}.metric b{font-size:22px}.gpu-panel{padding:12px}.gpu-readouts{grid-template-columns:minmax(0,1fr)}.gpu-shell{height:136px;padding-inline:12px}.gpu-fan{width:clamp(72px,25vw,92px)}.runtime{gap:8px 13px}.chart-head{align-items:start;flex-direction:column}}@media(prefers-reduced-motion:reduce){.gpu-fan-rotor{animation:none!important}}
.console-grid{grid-template-columns:minmax(0,var(--terminal-fit,720px)) minmax(320px,1fr)}.console-grid .terminal{overflow-x:hidden;white-space:pre;word-break:normal}@media(max-width:1050px){.console-grid{grid-template-columns:minmax(0,1fr)}}
.terminal-columns,.terminal-columns-mobile{height:25px;align-items:center;padding:0 14px 0 16px;background:#382f2a;color:#bfa183;border-left:4px solid var(--orange);border-bottom:1px solid #55463d;font:700 7px ui-monospace,Consolas,monospace;letter-spacing:.03em}.terminal-columns{display:grid;grid-template-columns:55px 46px 62px 62px 66px 48px minmax(80px,1fr) minmax(76px,1fr);gap:5px}.terminal-columns-mobile{display:none}.terminal-columns span,.terminal-columns-mobile span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}@media(max-width:560px){.terminal-columns{display:none}.terminal-columns-mobile{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}}
</style></head><body><header class="top"><div class="top-inner"><div class="brand"><small>K230 / STEEL BALL TRAINING TELEMETRY</small><h1>钢珠训练台</h1></div><div class="nav-row"><div class="segments"><button class="segment active" data-view="active">运行中 · <span id="activeCount">0</span></button><button class="segment" data-view="history">历史记录 · <span id="historyCount">0</span></button></div><span class="divider"></span><div class="models" id="models"></div></div><div class="cat-zone running" id="catZone"><span class="cat" aria-hidden="true">🐈</span></div></div></header><main id="main"><div class="empty"><div><strong>正在连接训练服务</strong><span>读取训练目录与实时日志…</span></div></div></main><script>
const state={payload:null,view:new URLSearchParams(location.search).get('view')||'active',run:new URLSearchParams(location.search).get('run'),rendered:null,anchor:null,terminalPinned:true,hoverIndex:null};
const esc=s=>String(s??'').replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));
const statusText={running:'训练中',starting:'启动中',completed:'已完成',failed:'训练异常',stopped:'已停止'};
const list=()=>state.payload?.[state.view]||[];
function chooseDefault(){const runs=list();if(!runs.some(r=>r.id===state.run))state.run=runs[0]?.id||null}
function updateUrl(){const q=new URLSearchParams({view:state.view});if(state.run)q.set('run',state.run);history.replaceState(null,'','?'+q)}
function renderNav(){document.querySelector('#activeCount').textContent=state.payload.active.length;document.querySelector('#historyCount').textContent=state.payload.history.length;document.querySelectorAll('.segment').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));const host=document.querySelector('#models');host.innerHTML=list().map(r=>`<button class="model-button ${r.id===state.run?'active':''}" data-run="${esc(r.id)}">${esc(r.name)}</button>`).join('');host.querySelectorAll('button').forEach(b=>b.onclick=()=>{state.run=b.dataset.run;updateUrl();renderNav();renderRun(true)})}
function runData(){return list().find(r=>r.id===state.run)}
function renderRun(force=false){const d=runData(),main=document.querySelector('#main');if(!d){state.rendered=null;main.innerHTML=`<div class="empty"><div><strong>${state.view==='active'?'当前没有正在训练的模型':'还没有历史训练'}</strong><span>${state.view==='active'?'完成后的训练会自动转入历史记录。':'启动训练后，这里会永久保留结果。'}</span></div></div>`;document.querySelector('#catZone').className='cat-zone completed';return}if(!force&&state.rendered===d.id){updateRun(d);return}state.rendered=d.id;main.innerHTML=`<section><div class="run-title"><div><h2>${esc(d.name)}</h2><p>庐山派 K230 钢珠检测候选模型 · ${esc(d.id)}</p></div><div class="status ${d.status}">${esc(statusText[d.status]||d.status)} / 更新 ${esc(d.updated_at)}</div></div><div class="progress-grid"><div class="progress-panel"><div class="progress-head"><div><strong id="epochLabel">0 / 0</strong><small>总体训练进度</small></div><span id="overallPercent">0.00%</span></div><div class="track" id="overallTrack"><div class="fill" id="overallFill"></div></div><div class="estimate">按已完成轮次与当前批次计算</div></div><div class="progress-panel"><div class="progress-head"><div><strong id="batchLabel">0 / 0</strong><small>当前轮次进度</small></div><span id="epochPercent">0.00%</span></div><div class="track" id="epochTrack"><div class="fill" id="epochFill"></div></div><div class="estimate">批次间平滑估算 · 不会提前跨过下一批次</div></div></div><div class="metrics"><div class="metric"><b id="map50">0.000</b><small>mAP50</small></div><div class="metric"><b id="precision">0.000</b><small>精确率</small></div><div class="metric"><b id="recall">0.000</b><small>召回率</small></div><div class="metric"><b id="plateau">0 / 7</b><small>平台期计数</small></div></div><div class="runtime"><span>GPU <b id="gpu">—</b></span><span>box loss <b id="boxLoss">0.0000</b></span><span>cls loss <b id="clsLoss">0.0000</b></span><span>速度 <b id="rate">—</b></span><span>ETA <b id="eta">—</b></span></div><div class="terminal-head"><span>训练终端 / LIVE OUTPUT</span><span class="heartbeat"><i></i><b id="heartbeat">接收中</b></span></div><pre class="terminal ${d.status==='failed'?'error':''}" id="terminal"></pre><section class="chart-section"><div class="chart-head"><h3>训练趋势</h3><div class="legend"><span><i style="background:#d87352"></i>mAP50</span><span><i style="background:#56866f"></i>精确率</span><span><i style="background:#bd8b2e"></i>召回率</span></div></div><div class="chart-box"><canvas id="chart"></canvas><div class="tooltip" id="tooltip"></div><div class="chart-empty" id="chartEmpty">完成下一轮后生成趋势曲线</div></div></section></section>`;const term=document.querySelector('#terminal');term.addEventListener('scroll',()=>{state.terminalPinned=term.scrollHeight-term.scrollTop-term.clientHeight<18});const canvas=document.querySelector('#chart');canvas.addEventListener('pointermove',chartHover);canvas.addEventListener('pointerleave',()=>document.querySelector('#tooltip').style.display='none');new ResizeObserver(()=>drawChart(runData())).observe(canvas);updateRun(d)}
function ensureGpuPanel(){const term=document.querySelector('#terminal');if(!term||document.querySelector('#gpuPanel'))return;const head=term.previousElementSibling;if(!head?.classList.contains('terminal-head'))return;const grid=document.createElement('div'),column=document.createElement('section');grid.className='console-grid';column.className='console-column';head.before(grid);column.append(head,term);grid.append(column);grid.insertAdjacentHTML('beforeend',`<aside class="gpu-panel unavailable" id="gpuPanel" aria-label="显卡实时状态"><div class="gpu-panel-head"><div><small>GRAPHICS PROCESSOR / LIVE TELEMETRY</small><strong id="gpuName">等待显卡数据</strong></div><span class="gpu-live" id="gpuLive">CONNECTING</span></div><div class="gpu-readouts"><div class="gpu-stat"><div class="gpu-stat-head"><span>GPU 利用率</span><b id="gpuUtil">—</b></div><div class="gpu-meter"><i id="gpuUtilBar"></i></div></div><div class="gpu-stat"><div class="gpu-stat-head"><span>显存利用率</span><b id="gpuMemoryPercent">—</b></div><div class="gpu-meter"><i id="gpuMemoryBar"></i></div></div></div><div class="gpu-shell" aria-hidden="true"><div class="gpu-fan"><i class="gpu-fan-rotor"></i></div><div class="gpu-fan"><i class="gpu-fan-rotor"></i></div></div><div class="gpu-foot"><span>VRAM <b id="gpuMemory">—</b></span><span>温度 <b id="gpuTemp">—</b></span><span>风扇 <b id="gpuFan">—</b></span></div></aside>`)}function updateGpuPanel(d){ensureGpuPanel();const panel=document.querySelector('#gpuPanel');if(!panel)return;const g=state.payload?.gpu||{available:false},write=(id,value)=>{const el=document.querySelector('#'+id);if(el)el.textContent=value},pct=value=>Math.max(0,Math.min(100,Number(value)||0));panel.classList.toggle('unavailable',!g.available);panel.classList.toggle('running',Boolean(g.available&&d?.status==='running'));if(!g.available){write('gpuName','未检测到 NVIDIA 遥测');write('gpuLive','NO SENSOR');write('gpuUtil','—');write('gpuMemoryPercent','—');write('gpuMemory','—');write('gpuTemp','—');write('gpuFan','—');document.querySelector('#gpuUtilBar').style.width='0%';document.querySelector('#gpuMemoryBar').style.width='0%';return}const util=pct(g.utilization),memory=pct(g.memory_percent),fan=pct(g.fan_percent),fanLoad=fan||Math.max(18,util);write('gpuName',`GPU ${g.index} · ${g.name}`);write('gpuLive',d?.status==='running'?'TRAINING / LIVE':'IDLE / HISTORY');write('gpuUtil',util.toFixed(0)+'%');write('gpuMemoryPercent',memory.toFixed(0)+'%');write('gpuMemory',(g.memory_used_mb/1024).toFixed(1)+' / '+(g.memory_total_mb/1024).toFixed(1)+' GB');write('gpuTemp',Number(g.temperature).toFixed(0)+' °C');write('gpuFan',fan?fan.toFixed(0)+'%':'自动');document.querySelector('#gpuUtilBar').style.width=util+'%';document.querySelector('#gpuMemoryBar').style.width=memory+'%';panel.style.setProperty('--fan-speed',Math.max(.38,2.25-fanLoad*.018).toFixed(2)+'s')}
function fitConsole(){const grid=document.querySelector('.console-grid'),term=document.querySelector('#terminal'),d=runData();if(!grid||!term||!d)return;if(!grid.dataset.fitWatch){grid.dataset.fitWatch='1';new ResizeObserver(()=>requestAnimationFrame(fitConsole)).observe(grid)}const raw=d.terminal||'',lines=raw.split('\n'),style=getComputedStyle(term),canvas=fitConsole.canvas||(fitConsole.canvas=document.createElement('canvas')),ctx=canvas.getContext('2d');ctx.font=style.fontSize+' '+style.fontFamily;const charWidth=Math.max(1,ctx.measureText('0').width),longest=lines.reduce((width,line)=>Math.max(width,ctx.measureText(line).width),0);if(matchMedia('(max-width:1050px)').matches){grid.style.removeProperty('--terminal-fit')}else{const maxWidth=Math.max(500,grid.clientWidth-334),target=Math.min(maxWidth,Math.max(500,Math.ceil(longest+38)));grid.style.setProperty('--terminal-fit',target+'px')}const capacity=Math.max(24,Math.floor((term.clientWidth-34)/charWidth)),rendered=lines.map(line=>line.length>capacity?line.slice(0,capacity-1)+'…':line).join('\n'),pinned=term.scrollHeight-term.scrollTop-term.clientHeight<20;if(term.textContent!==rendered)term.textContent=rendered;if(pinned)term.scrollTop=term.scrollHeight}
function ensureTerminalColumns(){const term=document.querySelector('#terminal');if(!term||term.previousElementSibling?.classList.contains('terminal-columns-mobile'))return;term.insertAdjacentHTML('beforebegin',`<div class="terminal-columns" aria-label="终端数据列"><span>EPOCH<br>轮次</span><span>VRAM<br>显存</span><span>BOX LOSS</span><span>CLS LOSS</span><span>DFL LOSS</span><span>INST<br>目标</span><span>SIZE · BATCH<br>输入 · 批次</span><span>SPEED · ETA<br>速度 · 剩余</span></div><div class="terminal-columns-mobile"><span>轮次 · 显存</span><span>训练损失</span><span>输入 · 批次</span><span>速度 · 剩余</span></div>`)}
function setText(id,value,pulse=false){if(id==='epochLabel'){updateGpuPanel(runData());ensureTerminalColumns();requestAnimationFrame(fitConsole)}const el=document.querySelector('#'+id);if(!el)return;if(el.textContent!==String(value)){el.textContent=value;if(pulse){el.parentElement.classList.remove('changed');void el.offsetWidth;el.parentElement.classList.add('changed')}}}
function updateRun(d){document.querySelector('#catZone').className='cat-zone '+d.status;setText('epochLabel',`${d.epoch.current} / ${d.epoch.total}`);setText('batchLabel',d.batch.total?`${d.batch.current} / ${d.batch.total}`:'等待批次');setText('map50',d.metrics.map50.toFixed(3),true);setText('precision',d.metrics.precision.toFixed(3),true);setText('recall',d.metrics.recall.toFixed(3),true);setText('plateau',`${d.plateau.stalled} / ${d.plateau.patience}`,true);setText('gpu',d.runtime.gpu_memory);setText('boxLoss',d.runtime.box_loss.toFixed(4));setText('clsLoss',d.runtime.cls_loss.toFixed(4));setText('rate',d.batch.rate?d.batch.rate.toFixed(1)+' it/s':'—');setText('eta',d.batch.eta);setText('heartbeat',d.status==='running'?'实时同步 '+state.payload.generated_at:(statusText[d.status]||d.status));const term=document.querySelector('#terminal');if(term&&term.textContent!==d.terminal){const pinned=state.terminalPinned;term.textContent=d.terminal;if(pinned)term.scrollTop=term.scrollHeight}state.anchor={at:performance.now(),data:d};document.querySelectorAll('.track').forEach(t=>t.classList.toggle('paused',d.status!=='running'));drawChart(d)}
function animateProgress(){const a=state.anchor;if(a&&document.querySelector('#epochFill')){const d=a.data,bt=d.batch.total,bc=d.batch.current;let epochPct=bt?bc/bt*100:0;if(d.status==='running'&&bt&&d.batch.rate&&d.log_age_seconds<5){const elapsed=Math.min((performance.now()-a.at)/1000,Math.max(0,5-d.log_age_seconds));const predicted=elapsed*d.batch.rate/bt*100;const cap=Math.max(epochPct,(bc+1)/bt*100-.001);epochPct=Math.min(epochPct+predicted,cap)}const overall=d.epoch.total?Math.min(100,((Math.max(0,d.epoch.current-1)+epochPct/100)/d.epoch.total)*100):0;setText('epochPercent',epochPct.toFixed(2)+'%');setText('overallPercent',overall.toFixed(2)+'%');document.querySelector('#epochFill').style.width=Math.max(0,Math.min(100,epochPct))+'%';document.querySelector('#overallFill').style.width=Math.max(0,Math.min(100,overall))+'%'}requestAnimationFrame(animateProgress)}
function drawChart(d){const c=document.querySelector('#chart');if(!c||!d)return;const box=c.parentElement,ratio=devicePixelRatio||1,w=box.clientWidth,h=box.clientHeight;c.width=w*ratio;c.height=h*ratio;const x=c.getContext('2d');x.scale(ratio,ratio);x.clearRect(0,0,w,h);const pad={l:46,r:22,t:24,b:34},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;x.font='10px ui-monospace,monospace';x.fillStyle='#7a6253';x.strokeStyle='rgba(150,116,86,.27)';x.lineWidth=1;for(let i=0;i<=4;i++){const y=pad.t+ph*i/4;x.beginPath();x.moveTo(pad.l,y);x.lineTo(w-pad.r,y);x.stroke();x.fillText((1-i*.25).toFixed(2),8,y+3)}const n=d.series.epochs.length;document.querySelector('#chartEmpty').style.display=n<2?'grid':'none';if(!n)return;const maxEpoch=Math.max(d.epoch.total,d.series.epochs[n-1],1),px=e=>pad.l+(e-1)/Math.max(1,maxEpoch-1)*pw,py=v=>pad.t+(1-Math.max(0,Math.min(1,v)))*ph;const datasets=[['map50','#d87352'],['precision','#56866f'],['recall','#bd8b2e']];datasets.forEach(([key,color])=>{const vals=d.series[key];x.strokeStyle=color;x.lineWidth=2;x.beginPath();vals.forEach((v,i)=>{const xx=px(d.series.epochs[i]),yy=py(v);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();x.fillStyle=color;vals.forEach((v,i)=>{x.beginPath();x.arc(px(d.series.epochs[i]),py(v),3.2,0,Math.PI*2);x.fill()})});const current=px(d.epoch.current);x.setLineDash([5,5]);x.strokeStyle='#9c432d';x.beginPath();x.moveTo(current,pad.t);x.lineTo(current,h-pad.b);x.stroke();x.setLineDash([]);x.fillStyle='#7a6253';[1,Math.ceil(maxEpoch/2),maxEpoch].forEach(e=>x.fillText(String(e),px(e)-3,h-12));c._chart={d,pad,pw,ph,w,h,px,py}}
function chartHover(ev){const c=ev.currentTarget,g=c._chart;if(!g||!g.d.series.epochs.length)return;const rect=c.getBoundingClientRect(),mx=ev.clientX-rect.left,epochs=g.d.series.epochs;let best=0,dist=Infinity;epochs.forEach((e,i)=>{const q=Math.abs(g.px(e)-mx);if(q<dist){dist=q;best=i}});const tip=document.querySelector('#tooltip');tip.innerHTML=`第 ${epochs[best]} 轮<br>mAP50　${g.d.series.map50[best].toFixed(4)}<br>精确率　${g.d.series.precision[best].toFixed(4)}<br>召回率　${g.d.series.recall[best].toFixed(4)}`;tip.style.display='block';tip.style.left=Math.min(g.w-135,Math.max(5,g.px(epochs[best])+10))+'px';tip.style.top='18px'}
async function refresh(){try{const payload=await fetch('/api?'+Date.now()).then(r=>{if(!r.ok)throw Error(r.status);return r.json()});state.payload=payload;chooseDefault();updateUrl();renderNav();renderRun()}catch(e){const hb=document.querySelector('#heartbeat');if(hb)hb.textContent='监控连接中断';document.querySelectorAll('.track').forEach(t=>t.classList.add('paused'))}}
document.querySelectorAll('.segment').forEach(b=>b.onclick=()=>{state.view=b.dataset.view;state.run=null;chooseDefault();updateUrl();renderNav();renderRun(true)});refresh();setInterval(refresh,1000);requestAnimationFrame(animateProgress);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api"):
            body = json.dumps(collect_runs(), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            body = HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("TRAINING_DASHBOARD=http://127.0.0.1:8765")
    server.serve_forever()
