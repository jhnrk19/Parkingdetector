"""Parkolóhely-foglaltság detektáló HTTP szerver (FastAPI, sima HTTP, 8000-es port).

Végpontok (a vékony klienssel egyező szerződés):
  GET  /api/health
  POST /api/train                 – 1. (tanító) videó; visszaadja a job_id-t és a session_id-t
  POST /api/analyze               – 2. (teszt) videó egy korábbi session_id tanult profiljával
  GET  /api/jobs/<job_id>         – állapot, haladás, statisztika
  GET  /api/jobs/<job_id>/video   – a feldolgozott MP4

Indítás:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import training_pipeline as tp
import vision_pipeline as vp

MODEL_NAME = "yolov8x-obb.pt"
DATA = Path(os.environ.get("PK_DATA", "./data")).expanduser().resolve()
JOBS_DIR, SESSIONS_DIR = DATA / "jobs", DATA / "sessions"
KEEP_HOURS = float(os.environ.get("PK_KEEP_HOURS", "72"))      # ennyi ideig maradnak meg a feltöltött és az eredményvideók
DEBUG_IMAGES = os.environ.get("PK_DEBUG", "1") != "0"          # háttér-, maszk- és helykép mentése a feladat mappájába
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts", ".mpg", ".mpeg", ".wmv"}

state: dict = {"model": None, "device": "cpu", "model_path": None, "load_error": None}
jobs: dict[str, dict] = {}
sessions: dict[str, dict] = {}
lock = threading.Lock()          # a jobs és a sessions szótár védelme
model_lock = threading.Lock()    # a YOLO következtetés egyszerre egy szálról fut (nincs CUDA-memóriahiba, versenyhelyzet)
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("PK_WORKERS", "2")), thread_name_prefix="pk")


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def find_model() -> str:
    """Keresési sorrend: YOLO_MODEL_PATH, ./yolov8x-obb.pt, ~/.cache/ultralytics/yolov8x-obb.pt, végül az Ultralytics letöltése."""
    env = os.environ.get("YOLO_MODEL_PATH")
    for cand in ([Path(env)] if env else []) + [Path.cwd() / MODEL_NAME, Path.home() / ".cache" / "ultralytics" / MODEL_NAME]:
        if cand.is_file():
            return str(cand)
    return MODEL_NAME


def load_model() -> None:
    import torch
    from ultralytics import YOLO
    state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    path = find_model()
    model = YOLO(path)
    try:
        model.to(state["device"])
    except Exception as e:  # noqa: BLE001 – ha a videókártya mégsem használható, processzoron megy tovább
        log(f"A modell nem tölthető a videókártyára ({e}); processzoron fut.")
        state["device"] = "cpu"
        model.to("cpu")
    state.update(model=model, model_path=path)
    log(f"Modell betöltve: {path}, eszköz: {state['device']}" + (f" ({torch.cuda.get_device_name(0)})" if state["device"] == "cuda" else ""))


def load_sessions() -> None:
    """A korábban megtanult profilok újraindítás után is használhatók."""
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            sessions[f.stem] = {"status": "ready", "profile": vp.ParkingConceptProfile.from_json(json.dumps(doc["profile"])), "created": doc.get("created", 0)}
        except Exception as e:  # noqa: BLE001
            log(f"Hibás munkamenetfájl kihagyva: {f.name} ({e})")
    if sessions:
        log(f"{len(sessions)} korábbi munkamenet betöltve.")


def cleanup_loop() -> None:
    while True:
        try:
            limit = time.time() - KEEP_HOURS * 3600
            for d in JOBS_DIR.iterdir():
                with lock:
                    job = jobs.get(d.name)
                    busy = job is not None and job["status"] in ("queued", "processing")
                if d.is_dir() and not busy and d.stat().st_mtime < limit:
                    shutil.rmtree(d, ignore_errors=True)
                    with lock:
                        jobs.pop(d.name, None)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1800)


@asynccontextmanager
async def lifespan(_: FastAPI):
    for d in (JOBS_DIR, SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    try:
        load_model()   # egyszer, induláskor – nem kérésenként
    except Exception as e:  # noqa: BLE001
        state["load_error"] = str(e)
        log(f"HIBA: a modell nem tölthető be: {e}")
    load_sessions()
    threading.Thread(target=cleanup_loop, daemon=True).start()
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Parkolóhely-foglaltság detektáló szerver", lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message, "detail": message}, status_code=status)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else "Hiba történt."
    return error(exc.status_code, {404: "A kért erőforrás nem található.", 405: "Ez a művelet itt nem engedélyezett."}.get(exc.status_code, msg)
                 if msg in ("Not Found", "Method Not Allowed") else msg)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    fields = {str(e["loc"][-1]) for e in exc.errors() if e.get("loc")}
    if "video" in fields:
        return error(400, "Hiányzó vagy üres 'video' fájl.")
    if "session_id" in fields:
        return error(400, "Hiányzó vagy üres 'session_id'.")
    return error(400, "Hibás kérés: " + ", ".join(sorted(fields)) if fields else "Hibás kérés.")


@app.exception_handler(Exception)
async def server_error(_: Request, exc: Exception):
    log("Váratlan hiba: " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
    return error(500, "Belső szerverhiba történt.")


# --------------------------------------------------------------------------- #
# Feladatok
# --------------------------------------------------------------------------- #

def new_job(mode: str, session_id: str) -> dict:
    job = {"job_id": uuid.uuid4().hex, "session_id": session_id, "mode": mode, "status": "queued", "progress": 0, "current_frame": 0,
           "total_frames": 0, "occupied": 0, "free": 0, "message": "Sorban áll.", "stats": None, "result_video_url": None, "error": None,
           "_video": None, "_created": time.time()}
    with lock:
        jobs[job["job_id"]] = job
    return job


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def save_upload(video: UploadFile | None, dest_dir: Path) -> Path | None:
    """A feltöltött videó lemezre mentése. None, ha hiányzik vagy üres."""
    if video is None or not (video.filename or "").strip():
        return None
    ext = Path(video.filename).suffix.lower()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("input" + (ext if ext in VIDEO_EXT else ".mp4"))
    with dest.open("wb") as out:
        shutil.copyfileobj(video.file, out, length=4 * 1024 * 1024)
    if dest.stat().st_size <= 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        return None
    return dest


def run_job(job_id: str, src: Path) -> None:
    with lock:
        job = jobs[job_id]
    mode, sid = job["mode"], job["session_id"]

    def progress(**kw) -> None:
        with lock:
            for k, v in kw.items():
                if k == "progress":
                    v = max(job["progress"], min(99, int(v)))   # a haladás nem megy visszafelé, a 100 csak a végén jár
                elif k in ("current_frame", "total_frames", "occupied", "free"):
                    v = max(0, int(v))
                job[k] = v

    try:
        if state["model"] is None:
            raise RuntimeError("A YOLO modell nincs betöltve: " + (state["load_error"] or "ismeretlen ok"))
        profile = None
        if mode == "test":
            with lock:
                profile = sessions[sid]["profile"]   # csak olvassuk: a munkamenet többször is felhasználható
        progress(status="processing", message="Feldolgozás indul…")
        log(f"{job_id[:8]} {mode}: indul ({src.name}, {src.stat().st_size / 1e6:.1f} MB)")
        t0 = time.time()
        out = src.parent / ("train_output.mp4" if mode == "train" else "test_output.mp4")
        detector = vp.Detector(state["model"], state["device"], model_lock)
        if mode == "train":
            def _vp_prog(**kw: object) -> None:
                if "progress" in kw and isinstance(kw["progress"], (int, float)):
                    kw["progress"] = min(38, int(float(kw["progress"]) * 0.38))
                progress(**kw)
            res = vp.process_video(src, out, detector, profile, _vp_prog, log=lambda m: log(f"{job_id[:8]} {m}"),
                                   debug_dir=src.parent / "debug" if DEBUG_IMAGES else None)
        else:
            res = vp.process_video(src, out, detector, profile, progress, log=lambda m: log(f"{job_id[:8]} {m}"),
                                   debug_dir=src.parent / "debug" if DEBUG_IMAGES else None)
        if not out.is_file() or out.stat().st_size <= 0:
            raise RuntimeError("A kimeneti videó nem jött létre.")
        report_url: str | None = None
        report_dir_str: str | None = None
        if mode == "train":
            prof = res["profile"]
            doc = {"session_id": sid, "created": time.time(), "profile": json.loads(prof.to_json())}
            tmp = SESSIONS_DIR / f"{sid}.json.tmp"
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, SESSIONS_DIR / f"{sid}.json")
            with lock:
                sessions[sid] = {"status": "ready", "profile": prof, "created": doc["created"]}

            log(f"{job_id[:8]} train: mélytanulási adat-előkészítés, transfer learning és riport indul…")
            tp.run_full_training_suite(
                video_path=src,
                job_dir=src.parent,
                base_model_path=state["model_path"] or find_model(),
                base_model=state["model"],
                device=state["device"],
                model_lock=model_lock,
                progress_cb=lambda pct, m: (
                    progress(status="processing", progress=pct, message=m),
                    log(f"{job_id[:8]} [{pct}%] {m}"),
                ),
            )
            report_dir_str = str(src.parent / "report")
            report_url = f"/api/jobs/{job_id}/report"
        total, occ = int(res["total_slots"]), int(res["final_occupied"])
        stats = {"total_slots": total, "final_occupied": occ, "final_free": total - occ, "frames_processed": int(res["frames_processed"])}
        msg = "Feldolgozás sikeresen befejezve."
        if total == 0:
            msg += " Parkolóhelyet nem találtam a videón."
        if res["notes"]:
            msg += " " + " ".join(res["notes"])
        with lock:   # a videó már a végleges helyén van: csak ezután lesz "completed"
            job.update(status="completed", progress=100, current_frame=stats["frames_processed"], total_frames=stats["frames_processed"],
                       occupied=occ, free=total - occ, stats=stats, message=msg, result_video_url=f"/api/jobs/{job_id}/video",
                       report_url=report_url, error=None, _video=str(out), _report_dir=report_dir_str)
        log(f"{job_id[:8]} {mode}: kész {time.time() - t0:.0f} mp alatt – összes {total}, foglalt {occ}, szabad {total - occ} "
            f"(csíkpár {res['sources']['stripes']}, pótolt {res['sources']['filled']}, járműből {res['sources']['vehicle']})")
    except Exception as e:  # noqa: BLE001
        log(f"{job_id[:8]} {mode}: HIBA – {e}\n{traceback.format_exc()}")
        with lock:
            job.update(status="failed", message="A feldolgozás sikertelen.", error=str(e) or type(e).__name__, stats=None, result_video_url=None)
            if mode == "train" and sessions.get(sid, {}).get("status") == "training":
                sessions[sid] = {"status": "failed", "profile": None, "created": time.time()}
    finally:
        try:
            src.unlink(missing_ok=True)   # a feltöltött eredeti nem kell tovább
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Végpontok
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "model_name": MODEL_NAME, "device": state["device"]}


@app.post("/api/train")
def train(video: UploadFile | None = File(None)):
    session_id = uuid.uuid4().hex
    job = new_job("train", session_id)
    src = save_upload(video, JOBS_DIR / job["job_id"])
    if src is None:
        with lock:
            jobs.pop(job["job_id"], None)
        return error(400, "Hiányzó vagy üres 'video' fájl.")
    with lock:
        sessions[session_id] = {"status": "training", "profile": None, "created": time.time()}
    executor.submit(run_job, job["job_id"], src)
    return JSONResponse({"job_id": job["job_id"], "session_id": session_id, "mode": "train", "status": "queued"}, status_code=202)


@app.post("/api/analyze")
def analyze(session_id: str | None = Form(None), video: UploadFile | None = File(None)):
    sid = (session_id or "").strip()
    if not sid:
        return error(400, "Hiányzó vagy üres 'session_id'.")
    with lock:
        sess = sessions.get(sid)
    if sess is None:
        return error(404, "Ismeretlen munkamenet-azonosító (session_id).")
    if sess["status"] != "ready" or sess["profile"] is None:
        return error(400, "A munkamenet tanítása még nem fejeződött be sikeresen." if sess["status"] == "training"
                     else "A munkamenet tanítása sikertelen volt; indíts új tanítást.")
    job = new_job("test", sid)
    src = save_upload(video, JOBS_DIR / job["job_id"])
    if src is None:
        with lock:
            jobs.pop(job["job_id"], None)
        return error(400, "Hiányzó vagy üres 'video' fájl.")
    executor.submit(run_job, job["job_id"], src)
    return JSONResponse({"job_id": job["job_id"], "session_id": sid, "mode": "test", "status": "queued"}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return error(404, "A megadott job_id nem található.")
        return public(job)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return error(404, "A megadott job_id nem található.")
        ready, path, mode = job["status"] == "completed", job["_video"], job["mode"]
    if not ready or not path or not Path(path).is_file() or Path(path).stat().st_size <= 0:
        return error(409, "A videó még nem készült el vagy a feldolgozás sikertelen volt.")
    name = "train_output.mp4" if mode == "train" else "test_output.mp4"
    # a FileResponse a fájl pontos méretét adja Content-Length-ként, és darabokban küldi
    return FileResponse(path, media_type="video/mp4", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return error(404, "A megadott job_id nem található.")
        ready, report_dir = job["status"] == "completed", job.get("_report_dir")
    if not ready or not report_dir:
        return error(409, "A tanítási riport még nem készült el vagy nem tanító feladatról van szó.")
    report_file = Path(report_dir) / "training_report.json"
    if not report_file.is_file():
        return error(404, "A tanítási riport fájl nem található.")
    return JSONResponse(json.loads(report_file.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/report/{image_name}")
def job_report_image(job_id: str, image_name: str):
    if "/" in image_name or "\\" in image_name or ".." in image_name or not image_name.endswith(".png"):
        return error(400, "Érvénytelen képfájlnév.")
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return error(404, "A megadott job_id nem található.")
        ready, report_dir = job["status"] == "completed", job.get("_report_dir")
    if not ready or not report_dir:
        return error(409, "A tanítási riport még nem készült el.")
    img_file = (Path(report_dir) / image_name).resolve()
    if img_file.parent != Path(report_dir).resolve() or not img_file.is_file():
        return error(404, "A kért ábra nem található.")
    return FileResponse(img_file, media_type="image/png")


@app.get("/api/sessions/{session_id}")
def session_info(session_id: str):
    """Kiegészítő végpont: a munkamenet tanult profilja (ellenőrzéshez; koordinátát nem tartalmaz)."""
    with lock:
        sess = sessions.get(session_id)
    if sess is None:
        return error(404, "Ismeretlen munkamenet-azonosító (session_id).")
    return {"session_id": session_id, "status": sess["status"], "profile": json.loads(sess["profile"].to_json()) if sess["profile"] else None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("PK_HOST", "0.0.0.0"), port=int(os.environ.get("PK_PORT", "8000")), log_level="warning")
