import yt_dlp
import cv2
import numpy as np
import collections
import threading
import time
import requests
import json
import queue
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

ACCIDENT_CONF_THRESHOLD = float(os.getenv("ACCIDENT_CONF_THRESHOLD", 0.35))
RECENT_CUTOFF           = int(os.getenv("RECENT_CUTOFF", 300))

def _get_footages_dir():
    from django.conf import settings
    return os.path.join(settings.BASE_DIR, "footages")

class _LiveSession:
    BUFFER_DURATION = 6.0

    def __init__(self):
        self.latest_frame          = None
        self.stream_active         = False
        self.t_first_frame_grabbed = None
        self.frame_grab_time       = None
        self.hiccup_count          = 0
        self.hiccup_ts             = None
        self.frame_buffer          = collections.deque()
        self.buffer_lock           = threading.Lock()

def _frame_grabber(session, stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        session.stream_active = False
        return

    first = True
    while session.stream_active:
        t_before   = time.perf_counter()
        ret, frame = cap.read()
        t_after    = time.perf_counter()

        if ret:
            session.frame_grab_time = (t_after - t_before) * 1000
            if first:
                session.t_first_frame_grabbed = time.perf_counter()
                first = False
            session.latest_frame = frame

            ts    = time.time()
            small = cv2.resize(frame, (640, 360))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
            jpg_bytes = buf.tobytes()

            with session.buffer_lock:
                session.frame_buffer.append((ts, jpg_bytes))
                cutoff = ts - _LiveSession.BUFFER_DURATION
                while session.frame_buffer and session.frame_buffer[0][0] < cutoff:
                    session.frame_buffer.popleft()
        else:
            session.hiccup_count += 1
            session.hiccup_ts     = time.perf_counter()
            time.sleep(0.5)
            ret2, frame2 = cap.read()
            if ret2:
                session.latest_frame = frame2
            else:
                session.stream_active = False
                break

    cap.release()

def _send_to_model(frame, model_server_url):
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = buffer.tobytes()

        t_start  = time.perf_counter()
        response = requests.post(
            f"{model_server_url.rstrip('/')}/detect",
            files={"image": ("frame.jpg", jpg_bytes, "image/jpeg")},
            timeout=15
        )
        roundtrip_ms = (time.perf_counter() - t_start) * 1000

        if response.status_code == 200:
            result = response.json()
            result["_roundtrip_ms"] = round(roundtrip_ms, 1)
            return result

        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0,
            "_roundtrip_ms": round(roundtrip_ms, 1),
            "_error": f"HTTP {response.status_code}"
        }

    except requests.exceptions.Timeout:
        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0, "_roundtrip_ms": -1,
            "_error": "Request timed out (>15s)"
        }
    except Exception as e:
        return {
            "detected": False, "detections": [], "false_positives": [],
            "inference_ms": 0, "_roundtrip_ms": -1,
            "_error": str(e)
        }

def _send_to_both_models(frame, accident_url, fire_url):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_accident = executor.submit(_send_to_model, frame, accident_url)
        future_fire     = executor.submit(_send_to_model, frame, fire_url)
        accident_result = future_accident.result()
        fire_result     = future_fire.result()
    return accident_result, fire_result

def _build_model_block(model_result):
    if model_result is None:
        return None

    if "_error" in model_result:
        return {"error": model_result["_error"]}

    inference_ms = model_result.get("inference_ms", 0)
    roundtrip_ms = model_result.get("_roundtrip_ms", 0)

    clean_detections = []
    for det in model_result.get("detections", []):
        x1, y1, x2, y2 = det["box"]
        clean_detections.append({
            "type":       det["type"],
            "confidence": det["confidence"],
            "coverage":   det.get("coverage", 0),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    clean_fp = []
    for fp in model_result.get("false_positives", []):
        x1, y1, x2, y2 = fp["box"]
        clean_fp.append({
            "type":       fp["type"],
            "confidence": fp["confidence"],
            "coverage":   fp.get("coverage", 0),
            "box":        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "fp_reason":  fp.get("fp_reason", "coverage exceeded threshold")
        })

    return {
        "detected":            model_result.get("detected", False),
        "detections":          clean_detections,
        "false_positives":     clean_fp,
        "inference_ms":        inference_ms,
        "roundtrip_ms":        roundtrip_ms,
        "network_overhead_ms": round(roundtrip_ms - inference_ms, 1)
    }

def _emit(payload):
    return json.dumps(payload) + "\n"

def _determine_incident_type(accident_block, fire_block):
    has_fire  = False
    has_smoke = False
    has_crash = False

    if fire_block and not fire_block.get("error") and fire_block.get("detected"):
        for det in fire_block.get("detections", []):
            det_type = det.get("type", "").lower()
            if "fire" in det_type:
                has_fire = True
            if "smoke" in det_type:
                has_smoke = True
        if not has_fire and not has_smoke:
            has_fire = True

    if accident_block and not accident_block.get("error") and accident_block.get("detected"):
        all_accident_detections = (
            accident_block.get("detections", []) +
            accident_block.get("false_positives", [])
        )
        for det in all_accident_detections:
            if det.get("confidence", 0) > ACCIDENT_CONF_THRESHOLD:
                has_crash = True
                break

    if not has_fire and not has_smoke and not has_crash:
        return None

    if has_crash and has_fire and has_smoke:
        return "cfs"
    if has_crash and has_fire:
        return "cf"
    if has_crash and has_smoke:
        return "cs"
    if has_fire and has_smoke:
        return "fs"
    if has_crash:
        return "c"
    if has_fire:
        return "f"
    if has_smoke:
        return "s"
    return "o"

def _save_snippet_live(session, incident_wall_time, footage_path):
    time.sleep(2.0)
    start_ts = incident_wall_time - 3.0
    end_ts   = incident_wall_time + 2.0
    with session.buffer_lock:
        snippet = [
            (ts, data)
            for ts, data in session.frame_buffer
            if start_ts <= ts <= end_ts
        ]
    if not snippet:
        return
    duration      = max(end_ts - start_ts, 1.0)
    estimated_fps = max(1.0, len(snippet) / duration)
    arr0        = np.frombuffer(snippet[0][1], np.uint8)
    first_frame = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
    if first_frame is None:
        return
    h, w = first_frame.shape[:2]
    os.makedirs(os.path.dirname(footage_path), exist_ok=True)
    writer = cv2.VideoWriter(
        footage_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        estimated_fps,
        (w, h)
    )
    for _, jpg_bytes in snippet:
        arr   = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            writer.write(frame)
    writer.release()

def _save_snippet_non_live(stream_url, incident_video_ts, footage_path):
    start_ts = max(0.0, incident_video_ts - 3.0)
    end_ts   = incident_video_ts + 2.0

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ts * 1000)
    
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return
    
    h, w = frame.shape[:2]
    os.makedirs(os.path.dirname(footage_path), exist_ok=True)
    writer = cv2.VideoWriter(
        footage_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    
    writer.write(frame)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        current_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_ts > end_ts:
            break
            
        writer.write(frame)
        
    writer.release()
    cap.release()

def _handle_camera_incident(camera_id, incident_type, session_or_url, incident_time, is_live=True):
    from .models import Camera, Camera_Incident
    from django.utils import timezone

    try:
        cam_obj = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        return {"error": f"Camera {camera_id} not found in database."}

    recent_cutoff = timezone.now() - datetime.timedelta(seconds=RECENT_CUTOFF)
    existing = Camera_Incident.objects.filter(
        camera=cam_obj,
        date_created__gte=recent_cutoff
    ).first()

    if existing:
        return {
            "created":       False,
            "id":            existing.id,
            "incident_type": existing.incident_type,
            "footage_path":  existing.footage
        }

    footages_dir     = _get_footages_dir()
    os.makedirs(footages_dir, exist_ok=True)
    incident_wall_time = time.time()
    footage_filename = f"cam_{camera_id}_{int(incident_wall_time)}.mp4"
    footage_path     = os.path.join(footages_dir, footage_filename)

    previous = Camera_Incident.objects.filter(camera=cam_obj).first()

    if previous:
        previous.incident_type = incident_type
        previous.date_created  = timezone.now()
        previous.footage       = footage_path
        previous.save()
        ci = previous
    else:
        ci = Camera_Incident.objects.create(
            camera        = cam_obj,
            incident_type = incident_type,
            date_created  = timezone.now(),
            footage       = footage_path
        )

    if is_live:
        threading.Thread(
            target=_save_snippet_live,
            args=(session_or_url, incident_time, footage_path),
            daemon=True
        ).start()
    else:
        threading.Thread(
            target=_save_snippet_non_live,
            args=(session_or_url, incident_time, footage_path),
            daemon=True
        ).start()

    return {
        "created":       True,
        "id":            ci.id,
        "incident_type": incident_type,
        "footage_path":  footage_path
    }

def _generate_non_live(stream_url, title, time_quantum, accident_url, fire_url):
    t_open  = time.perf_counter()
    cap     = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        yield _emit({"status": "error", "message": "Could not open video URL."})
        return
    open_ms = (time.perf_counter() - t_open) * 1000

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0

    yield _emit({
        "status":       "video_info",
        "title":        title,
        "duration_s":   round(duration_s, 2),
        "fps":          round(fps, 2),
        "total_frames": total_frames,
        "time_quantum": time_quantum,
        "timing":       {"video_open_ms": round(open_ms, 1)}
    })

    timestamps = []
    t = 0.0
    while t < duration_s:
        timestamps.append(t)
        t += time_quantum

    last_frame_time = (total_frames - 1) / fps if fps > 0 else 0
    if not timestamps or abs(timestamps[-1] - last_frame_time) > 0.01:
        timestamps.append(last_frame_time)

    frame_count = 0

    for i, ts in enumerate(timestamps):
        is_first = (i == 0)
        is_last  = (i == len(timestamps) - 1)

        t_seek     = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        seek_ms    = (time.perf_counter() - t_seek) * 1000

        if not ret or frame is None:
            yield _emit({
                "status":    "warning",
                "message":   f"Could not read frame at {ts:.2f}s — skipping.",
                "timestamp": ts
            })
            continue

        frame_count += 1

        if is_first and is_last:
            position_label = "first_and_last"
        elif is_first:
            position_label = "first"
        elif is_last:
            position_label = "last_edge_case"
        else:
            position_label = "mid"

        accident_result, fire_result = _send_to_both_models(frame, accident_url, fire_url)

        try:
            yield _emit({
                "status":      "frame",
                "frame_count": frame_count,
                "timestamp_s": round(ts, 2),
                "position":    position_label,
                "timing":      {"seek_and_read_ms": round(seek_ms, 1)},
                "models": {
                    "accident": _build_model_block(accident_result),
                    "fire":     _build_model_block(fire_result)
                }
            })
        except GeneratorExit:
            cap.release()
            return

        if not is_last:
            time.sleep(time_quantum)

    cap.release()
    yield _emit({
        "status":      "video_complete",
        "message":     "Video playback complete.",
        "frame_count": frame_count
    })

def _generate_live(stream_url, title, fetch_ms, time_quantum, accident_url, fire_url):
    session                = _LiveSession()
    session.stream_active  = True
    last_seen_hiccup_count = 0

    t_thread_start  = time.perf_counter()
    grabber         = threading.Thread(
        target=_frame_grabber,
        args=(session, stream_url),
        daemon=True
    )
    grabber.start()
    thread_spawn_ms = (time.perf_counter() - t_thread_start) * 1000

    t_wait = time.perf_counter()
    while session.latest_frame is None and session.stream_active:
        time.sleep(0.05)
        if time.perf_counter() - t_wait > 15:
            yield _emit({"status": "error", "message": "No frame received within 15s."})
            session.stream_active = False
            return

    if not session.stream_active:
        yield _emit({
            "status":      "stream_ended",
            "message":     "Live stream ended before first frame could be captured.",
            "frame_count": 0
        })
        return

    t_first_in_python = time.perf_counter()
    wait_ms     = (t_first_in_python - t_thread_start) * 1000
    grab_lag_ms = (session.t_first_frame_grabbed - t_thread_start) * 1000 \
                  if session.t_first_frame_grabbed else 0

    yield _emit({
        "status": "connected",
        "title":  title,
        "timing": {
            "fetch_ms":                round(fetch_ms, 1),
            "thread_spawn_ms":         round(thread_spawn_ms, 1),
            "buffer_fill_ms":          round(grab_lag_ms, 1),
            "poll_overhead_ms":        round(wait_ms - grab_lag_ms, 1),
            "total_to_first_frame_ms": round(wait_ms + fetch_ms, 1)
        }
    })

    frame_count = 0

    try:
        while True:
            if session.hiccup_count != last_seen_hiccup_count:
                last_seen_hiccup_count = session.hiccup_count
                yield _emit({
                    "status":       "hiccup",
                    "message":      "Stream read failed momentarily — attempting to continue.",
                    "hiccup_count": session.hiccup_count
                })

            if not session.stream_active:
                yield _emit({
                    "status":      "stream_ended",
                    "message":     "The live stream has ended.",
                    "frame_count": frame_count
                })
                return

            if session.latest_frame is not None:
                frame_count    += 1
                current_grab_ms = session.frame_grab_time or 0

                accident_result, fire_result = _send_to_both_models(
                    session.latest_frame, accident_url, fire_url
                )

                yield _emit({
                    "status":      "frame",
                    "frame_count": frame_count,
                    "timing":      {"cap_read_ms": round(current_grab_ms, 1)},
                    "models": {
                        "accident": _build_model_block(accident_result),
                        "fire":     _build_model_block(fire_result)
                    }
                })

            time.sleep(time_quantum)

    except GeneratorExit:
        session.stream_active = False
        return

def generate_stream_detections(youtube_url, live=True, time_quantum=3,
                                accident_model_url=None, fire_model_url=None):
    yield _emit({
        "status":       "initialising",
        "mode":         "live" if live else "non_live",
        "time_quantum": time_quantum
    })

    ydl_opts = {
        "format": "best" if live else "best[ext=mp4]/best",
        "quiet":  True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(youtube_url, download=False)
            stream_url = info["url"]
            title      = info.get("title", "Unknown")
    except Exception as e:
        yield _emit({"status": "error", "message": f"Could not fetch stream — {e}"})
        return

    t_fetch_start = time.perf_counter()
    fetch_ms      = (time.perf_counter() - t_fetch_start) * 1000

    yield _emit({
        "status": "url_fetched",
        "title":  title,
        "timing": {"fetch_ms": round(fetch_ms, 1)}
    })

    if not live:
        yield from _generate_non_live(
            stream_url, title, time_quantum, accident_model_url, fire_model_url
        )
    else:
        yield from _generate_live(
            stream_url, title, fetch_ms, time_quantum, accident_model_url, fire_model_url
        )

def _run_camera_with_incidents(camera, accident_url, fire_url, time_quantum,
                                event_queue, stop_event):
    camera_id  = camera["id"]
    camera_lat = camera["latitude"]
    camera_lon = camera["longitude"]
    camera_url = camera["url"]

    camera_tag = {
        "camera_id":        camera_id,
        "camera_latitude":  camera_lat,
        "camera_longitude": camera_lon,
        "camera_url":       camera_url
    }

    def put(payload):
        payload.update(camera_tag)
        event_queue.put(payload)

    put({"status": "initialising", "mode": "live", "time_quantum": time_quantum})

    ydl_opts = {"format": "best", "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(camera_url, download=False)
            stream_url = info["url"]
            title      = info.get("title", "Unknown")
    except Exception as e:
        put({"status": "error", "message": f"Could not fetch stream — {e}"})
        return

    put({"status": "url_fetched", "title": title})

    session               = _LiveSession()
    session.stream_active = True
    last_seen_hiccup      = 0

    grabber = threading.Thread(
        target=_frame_grabber,
        args=(session, stream_url),
        daemon=True
    )
    grabber.start()

    t_wait = time.perf_counter()
    while session.latest_frame is None and session.stream_active:
        time.sleep(0.05)
        if time.perf_counter() - t_wait > 15:
            put({"status": "error", "message": "No frame received within 15s."})
            session.stream_active = False
            return

    if not session.stream_active:
        put({"status": "stream_ended",
             "message": "Stream ended before first frame could be captured.",
             "frame_count": 0})
        return

    put({"status": "connected", "title": title})

    frame_count = 0

    try:
        while not stop_event.is_set():
            if session.hiccup_count != last_seen_hiccup:
                last_seen_hiccup = session.hiccup_count
                put({
                    "status":       "hiccup",
                    "message":      "Stream read failed momentarily — attempting to continue.",
                    "hiccup_count": session.hiccup_count
                })

            if not session.stream_active:
                put({"status": "stream_ended",
                     "message": "The live stream has ended.",
                     "frame_count": frame_count})
                return

            if session.latest_frame is None:
                time.sleep(time_quantum)
                continue

            frame_count    += 1
            current_grab_ms = session.frame_grab_time or 0

            accident_result, fire_result = _send_to_both_models(
                session.latest_frame, accident_url, fire_url
            )

            accident_block = _build_model_block(accident_result)
            fire_block     = _build_model_block(fire_result)

            incident_type = _determine_incident_type(accident_block, fire_block)

            camera_incident_info = None
            if incident_type:
                incident_wall_time   = time.time()
                camera_incident_info = _handle_camera_incident(
                    camera_id, incident_type, session, incident_wall_time, is_live=True
                )

            event = {
                "status":      "frame",
                "frame_count": frame_count,
                "timing":      {"cap_read_ms": round(current_grab_ms, 1)},
                "models": {
                    "accident": accident_block,
                    "fire":     fire_block
                }
            }

            if camera_incident_info:
                event["camera_incident"] = camera_incident_info

            put(event)
            time.sleep(time_quantum)

    except Exception as e:
        put({"status": "camera_error", "message": str(e)})
    finally:
        session.stream_active = False

def _run_camera_non_live(camera, accident_url, fire_url, time_quantum,
                          event_queue, stop_event):
    camera_id  = camera["id"]
    camera_lat = camera["latitude"]
    camera_lon = camera["longitude"]
    camera_url = camera["url"]

    camera_tag = {
        "camera_id":        camera_id,
        "camera_latitude":  camera_lat,
        "camera_longitude": camera_lon,
        "camera_url":       camera_url
    }

    def put(payload):
        payload.update(camera_tag)
        event_queue.put(payload)

    put({"status": "initialising", "mode": "non_live", "time_quantum": time_quantum})

    ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info       = ydl.extract_info(camera_url, download=False)
            stream_url = info["url"]
            title      = info.get("title", "Unknown")
    except Exception as e:
        put({"status": "error", "message": f"Could not fetch stream — {e}"})
        return

    put({"status": "url_fetched", "title": title})

    t_open = time.perf_counter()
    cap    = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        put({"status": "error", "message": "Could not open video URL."})
        return
    open_ms = (time.perf_counter() - t_open) * 1000

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0

    put({
        "status":       "video_info",
        "title":        title,
        "duration_s":   round(duration_s, 2),
        "fps":          round(fps, 2),
        "total_frames": total_frames,
        "time_quantum": time_quantum,
        "timing":       {"video_open_ms": round(open_ms, 1)}
    })

    timestamps = []
    t = 0.0
    while t < duration_s:
        timestamps.append(t)
        t += time_quantum

    last_frame_time = (total_frames - 1) / fps if fps > 0 else 0
    if not timestamps or abs(timestamps[-1] - last_frame_time) > 0.01:
        timestamps.append(last_frame_time)

    frame_count = 0

    for i, ts in enumerate(timestamps):
        if stop_event.is_set():
            break

        is_first = (i == 0)
        is_last  = (i == len(timestamps) - 1)

        t_seek     = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        seek_ms    = (time.perf_counter() - t_seek) * 1000

        if not ret or frame is None:
            put({
                "status":    "warning",
                "message":   f"Could not read frame at {ts:.2f}s — skipping.",
                "timestamp": ts
            })
            continue

        frame_count += 1

        if is_first and is_last:
            position_label = "first_and_last"
        elif is_first:
            position_label = "first"
        elif is_last:
            position_label = "last_edge_case"
        else:
            position_label = "mid"

        accident_result, fire_result = _send_to_both_models(frame, accident_url, fire_url)

        accident_block = _build_model_block(accident_result)
        fire_block     = _build_model_block(fire_result)

        incident_type = _determine_incident_type(accident_block, fire_block)

        camera_incident_info = None
        if incident_type:
            camera_incident_info = _handle_camera_incident(
                camera_id, incident_type, stream_url, ts, is_live=False
            )

        event = {
            "status":      "frame",
            "frame_count": frame_count,
            "timestamp_s": round(ts, 2),
            "position":    position_label,
            "timing":      {"seek_and_read_ms": round(seek_ms, 1)},
            "models": {
                "accident": accident_block,
                "fire":     fire_block
            }
        }

        if camera_incident_info:
            event["camera_incident"] = camera_incident_info

        put(event)

        if not is_last:
            time.sleep(time_quantum)

    cap.release()
    put({
        "status":      "video_complete",
        "message":     "Video playback complete.",
        "frame_count": frame_count
    })

def generate_multi_camera_stream(cameras, accident_model_url, fire_model_url, time_quantum, live=True):
    if not cameras:
        yield _emit({"status": "no_cameras", "message": "No cameras found in database."})
        return

    event_queue = queue.Queue()
    stop_event  = threading.Event()

    yield _emit({
        "status":       "initialising_all",
        "camera_count": len(cameras),
        "camera_ids":   [c["id"] for c in cameras]
    })

    threads = []
    for camera in cameras:
        target = _run_camera_with_incidents if live else _run_camera_non_live
        t = threading.Thread(
            target=target,
            args=(camera, accident_model_url, fire_model_url,
                  time_quantum, event_queue, stop_event),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        while True:
            try:
                payload = event_queue.get(timeout=1.0)
                yield json.dumps(payload) + "\n"
            except queue.Empty:
                if all(not t.is_alive() for t in threads):
                    yield _emit({
                        "status":  "all_cameras_ended",
                        "message": "All camera streams have ended."
                    })
                    return
                continue

    except GeneratorExit:
        stop_event.set()
        return