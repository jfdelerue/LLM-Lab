from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests
import streamlit as st
from faster_whisper import WhisperModel

MIN_IMAGE_LARGEST_SIDE_PX = 32
DEFAULT_THUMBNAIL_LARGEST_SIDE_PX = 400
MAX_THUMBNAIL_LARGEST_SIDE_PX = 1600
RUNS_DIR = Path("video_llm_lab_runs")
PROTOCOL_RUNS_DIR = Path("ollama_protocol_tests")
SETTINGS_PATH = Path(os.environ.get("VIDEO_LLM_LAB_SETTINGS", "video_llm_lab_settings.json"))


@dataclass
class Thumbnail:
    index: int
    timestamp_sec: float
    path: Path
    width: int
    height: int
    jpeg_bytes: int
    base64_bytes: int | None = None


class OllamaError(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


ProgressCallback = Callable[[float, str], None]


class ProgressReporter:
    """Render one task progress bar in the page-level placeholder."""

    def __init__(self, placeholder: Any, label: str) -> None:
        self.placeholder = placeholder
        self.label = label
        self.update(0.0, "Démarrage…")

    def update(self, fraction: float, detail: str = "") -> None:
        percent = max(0, min(100, round(fraction * 100)))
        self.placeholder.empty()
        with self.placeholder.container():
            st.caption(f"**{self.label}** — {detail or f'{percent} %'}")
            st.progress(percent, text=f"{percent} %")

    def complete(self, detail: str = "Terminé") -> None:
        self.update(1.0, detail)

    def fail(self, detail: str) -> None:
        self.placeholder.empty()
        with self.placeholder.container():
            st.error(f"{self.label} — {detail}")
            st.progress(0, text="Interrompu")


def default_settings() -> dict[str, Any]:
    return {
        "ollama_base_url": "http://localhost:11434", "ollama_model": "qwen2.5vl:7b",
        "ollama_num_ctx": 8192, "ollama_num_predict": 1024, "ollama_temperature": 0.0,
        "ollama_top_p": 0.9, "ollama_num_batch": 512,
        "thumbnail_largest_side_px": DEFAULT_THUMBNAIL_LARGEST_SIDE_PX, "thumbnail_interval_sec": 2.0,
        "thumbnail_max_frames": 48, "thumbnail_jpeg_quality": 85,
        "thumbnail_gallery_display_width": 400, "thumbnail_gallery_max_items": 24,
        "video_display_max_side": 400,
        "whisper_model_size": "small", "whisper_device": "auto", "whisper_compute_type": "int8",
        "whisper_language": "ru", "whisper_fallback_cpu": True,
        "analysis_language": "fr", "dialogue_language": "ru",
        "analysis_objective": "comprendre de quoi discute la personne et déterminer si l’image ajoute du contexte utile",
        "llm_batch_size": 4, "transcript_context_max_chars": 6000,
        "two_pass_max_keyframes": 12, "two_pass_high_quality_largest_side_px": 1280,
        "two_pass_high_quality_jpeg_quality": 90, "two_pass_context_before_sec": 0.0,
        "two_pass_context_after_sec": 0.0,
    }


def load_settings() -> dict[str, Any]:
    s = default_settings()
    if SETTINGS_PATH.exists():
        s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    return s


def save_settings(s: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def resize_by_largest_side(image: np.ndarray, largest_side_px: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(w, h) <= largest_side_px:
        return image
    scale = largest_side_px / max(w, h)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def fmt_ts(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000)); total = int(seconds)
    h, rem = divmod(total, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps, "frame_count": frames, "duration_sec": frames / fps if fps else 0}
    cap.release(); return meta


def extract_frame_at(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp) * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def save_jpeg(frame: np.ndarray, path: Path, quality: int, largest_side: int) -> Thumbnail:
    img = resize_by_largest_side(frame, largest_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: raise RuntimeError(f"Échec écriture JPEG: {path}")
    h, w = img.shape[:2]
    return Thumbnail(0, 0.0, path, w, h, path.stat().st_size)


def extract_thumbnails(video_path: Path, out_dir: Path, interval: float, max_frames: int, largest_side: int, quality: int,
                       progress: ProgressCallback | None = None) -> list[Thumbnail]:
    cap = cv2.VideoCapture(str(video_path)); meta = video_metadata(video_path)
    thumbs = []
    for i in range(max_frames):
        ts = i * interval
        if meta["duration_sec"] and ts > meta["duration_sec"]: break
        frame = extract_frame_at(cap, ts)
        if frame is None: break
        p = out_dir / f"thumb_{i+1:06d}_{ts:.2f}s.jpg"
        t = save_jpeg(frame, p, quality, largest_side); t.index = i + 1; t.timestamp_sec = ts; thumbs.append(t)
        if progress: progress((i + 1) / max_frames, f"Vignette {i + 1}/{max_frames}")
    cap.release(); return thumbs


def extract_keyframes(video_path: Path, selected: list[dict[str, Any]], out_dir: Path, max_items: int, largest_side: int, quality: int, before: float, after: float,
                      progress: ProgressCallback | None = None) -> list[Thumbnail]:
    cap = cv2.VideoCapture(str(video_path)); out=[]; seen=set(); idx=1
    selected_items = selected[:max_items]
    for item_number, item in enumerate(selected_items, 1):
        ts0 = float(item.get("timestamp_sec", 0))
        for ts in [ts0 - before, ts0, ts0 + after]:
            ts = max(0.0, ts)
            key = round(ts, 2)
            if key in seen: continue
            seen.add(key); frame = extract_frame_at(cap, ts)
            if frame is None: continue
            p = out_dir / f"keyframe_{idx:06d}_{ts:.2f}s.jpg"
            t = save_jpeg(frame, p, quality, largest_side); t.index=idx; t.timestamp_sec=ts; out.append(t); idx += 1
        if progress: progress(item_number / max(1, len(selected_items)), f"Sélection {item_number}/{len(selected_items)}")
    cap.release(); return out


def thumbnail_to_ollama_base64(thumbnail: Thumbnail, jpeg_quality: int = 85) -> tuple[str, dict[str, Any]]:
    img = cv2.imread(str(thumbnail.path), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Image illisible: {thumbnail.path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok: raise RuntimeError("Réencodage JPEG impossible.")
    raw = bytes(buf); b64 = base64.b64encode(raw).decode("ascii")
    h, w = rgb.shape[:2]
    return b64, {"width": w, "height": h, "jpeg_bytes": len(raw), "base64_bytes": len(b64)}


def compute_image_payload_report(thumbnails: list[Thumbnail], batch_size: int) -> list[dict[str, Any]]:
    rows=[]
    for i,t in enumerate(thumbnails):
        b64_len = t.base64_bytes or len(base64.b64encode(t.path.read_bytes()))
        rows.append({"index": t.index, "timestamp": round(t.timestamp_sec, 3), "path": str(t.path), "width": t.width,
                     "height": t.height, "largest_side": max(t.width, t.height), "jpeg_kb": round(t.jpeg_bytes/1024,1),
                     "base64_kb": round(b64_len/1024,1), "batch_id": i // max(1,batch_size) + 1})
    return rows


def raise_ollama_error(response: requests.Response, context: str) -> None:
    raise OllamaError(f"{context}\nHTTP {response.status_code}\nRéponse Ollama:\n{response.text[:3000]}")


def ollama_list_models(base_url: str, timeout: float = 5) -> list[str]:
    """Return locally installed Ollama model names in server order."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama inaccessible: {exc}") from exc
    if not response.ok:
        raise_ollama_error(response, "Échec du scan des modèles Ollama (/api/tags)")
    try:
        return [model["name"] for model in response.json().get("models", []) if model.get("name")]
    except (TypeError, ValueError) as exc:
        raise OllamaError("Réponse /api/tags invalide.") from exc


def ollama_model_capabilities(base_url: str, model: str, timeout: float = 30) -> list[str]:
    """Read the capabilities advertised by the installed model via /api/show."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/show", json={"model": model}, timeout=timeout
        )
    except requests.RequestException as exc:
        raise OllamaError(f"Impossible d'inspecter le modèle {model}: {exc}") from exc
    if not response.ok:
        raise_ollama_error(response, f"Échec de l'inspection de {model} (/api/show)")
    try:
        return [str(item) for item in response.json().get("capabilities", [])]
    except (TypeError, ValueError) as exc:
        raise OllamaError("Réponse /api/show invalide.") from exc


def require_vision_model(s: dict[str, Any]) -> None:
    capabilities = ollama_model_capabilities(s["ollama_base_url"], s["ollama_model"])
    if "vision" not in capabilities:
        raise OllamaError(
            f"Le modèle {s['ollama_model']} n'annonce pas la capacité « vision » dans /api/show. "
            "Utilisez ce modèle pour B (texte), ou sélectionnez un modèle vision pour A/C."
        )


def redact_images(value: Any) -> Any:
    """Keep a useful, compact trace without duplicating large base64 images."""
    if isinstance(value, dict):
        return {key: ([{"base64_bytes": len(item), "sha256": hashlib.sha256(item.encode()).hexdigest()} for item in child]
                      if key == "images" and isinstance(child, list) else redact_images(child))
                for key, child in value.items()}
    if isinstance(value, list):
        return [redact_images(item) for item in value]
    return value


def ollama_protocol_test(endpoint: str, prompt: str, s: dict[str, Any], images: list[str] | None = None) -> dict[str, Any]:
    """Call Ollama without normalising its answer and build a reproducible trace."""
    if endpoint == "chat":
        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        payload = {"model": s["ollama_model"], "messages": [message], "stream": False,
                   "options": ollama_options(s)}
    else:
        payload = {"model": s["ollama_model"], "prompt": prompt, "stream": False,
                   "options": ollama_options(s)}
        if images:
            payload["images"] = images
    url = f"{s['ollama_base_url'].rstrip('/')}/api/{endpoint}"
    started = datetime.now(timezone.utc)
    try:
        response = requests.post(url, json=payload, timeout=900)
    except requests.RequestException as exc:
        raise OllamaError(f"Échec du test de protocole: {exc}") from exc
    finished = datetime.now(timezone.utc)
    try:
        parsed_response: Any = response.json()
    except ValueError:
        parsed_response = None
    report = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "duration_ms": round((finished - started).total_seconds() * 1000, 3),
        "ollama": {"url": url, "endpoint": endpoint, "model": s["ollama_model"]},
        "request": redact_images(payload),
        "http": {"status_code": response.status_code, "headers": dict(response.headers)},
        # Both forms matter when investigating model-specific <think> markers or fields.
        "response_json": parsed_response,
        "response_raw": response.text,
    }
    if not response.ok:
        report["error"] = "Réponse HTTP Ollama non réussie"
    return report


def ollama_options(s: dict[str, Any]) -> dict[str, Any]:
    return {"num_ctx": s["ollama_num_ctx"], "num_predict": s["ollama_num_predict"], "temperature": s["ollama_temperature"], "top_p": s["ollama_top_p"], "num_batch": s["ollama_num_batch"]}


def append_request_log(request_log: list[dict[str, Any]] | None, endpoint: str,
                       payload: dict[str, Any]) -> dict[str, Any] | None:
    if request_log is not None:
        entry = {
            "url": endpoint,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "payload": redact_images(payload),
        }
        request_log.append(entry)
        return entry
    return None


def complete_request_log(entry: dict[str, Any] | None, response: requests.Response) -> None:
    """Attach the complete, unmodified HTTP response to an existing request trace."""
    if entry is not None:
        entry.update({
            "received_at": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status_code,
            "response_headers": dict(response.headers),
            "response_raw": response.text,
        })


def ollama_chat_text(prompt: str, s: dict[str, Any],
                     request_log: list[dict[str, Any]] | None = None) -> str:
    """Send text through /api/chat so Ollama applies each model's chat template."""
    payload = {"model": s["ollama_model"], "messages": [{"role": "user", "content": prompt}],
               "stream": False, "options": ollama_options(s)}
    url = f"{s['ollama_base_url'].rstrip('/')}/api/chat"
    log_entry = append_request_log(request_log, url, payload)
    r = requests.post(url, json=payload, timeout=600)
    complete_request_log(log_entry, r)
    if not r.ok: raise_ollama_error(r, "Échec Ollama texte /api/chat")
    return r.json().get("message", {}).get("content", "")


def ollama_generate(prompt: str, s: dict[str, Any],
                    request_log: list[dict[str, Any]] | None = None) -> str:
    """Backward-compatible text helper used outside the A/B/C workflow."""
    return ollama_chat_text(prompt, s, request_log)


def ollama_chat_vision(prompt: str, thumbs: list[Thumbnail], s: dict[str, Any], quality: int,
                       request_log: list[dict[str, Any]] | None = None) -> str:
    images=[thumbnail_to_ollama_base64(t, quality)[0] for t in thumbs]
    payload={"model": s["ollama_model"], "messages":[{"role":"user","content":prompt,"images":images}], "stream":False, "options":ollama_options(s)}
    url=f"{s['ollama_base_url'].rstrip('/')}/api/chat"
    log_entry = append_request_log(request_log, url, payload)
    r=requests.post(url, json=payload, timeout=900)
    complete_request_log(log_entry, r)
    if r.ok: return r.json().get("message",{}).get("content","")
    if len(thumbs) > 1:
        parts=[]
        for t in thumbs:
            parts.append(ollama_chat_vision(prompt + f"\nImage unique: index {t.index}, timestamp {t.timestamp_sec:.2f}s", [t], s, quality, request_log))
        return "\n\n".join(parts)
    raise_ollama_error(r, "Échec Ollama vision /api/chat")


def build_reduced_transcript_context(transcript: str, max_chars: int) -> str:
    if len(transcript) <= max_chars: return transcript
    half=max_chars//2
    return transcript[:half] + "\n...[transcript réduit]...\n" + transcript[-half:]


def extract_json_from_text(text: str) -> dict | None:
    for candidate in [text, *re.findall(r"\{.*\}", text, flags=re.S)]:
        try: return json.loads(candidate)
        except Exception: pass
    return None



def trim_text_middle(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...[contenu réduit pour rester dans le contexte]...\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def is_placeholder_llm_response(text: str) -> bool:
    cleaned = re.sub(r"[\s#*_`>\-–—:.;,!？?]+", "", text or "")
    return len(cleaned) < 12


def build_comparison_prompt(s: dict[str, Any], analyses: dict[str, str]) -> str:
    max_total = max(1000, int(s["transcript_context_max_chars"]))
    per_section = max(800, max_total // max(1, len(analyses)))
    sections = []
    for key, label in [("A", "Images seules"), ("B", "Transcript seul"), ("C", "Images + transcript"), ("D", "Two-pass keyframes")]:
        content = trim_text_middle(analyses.get(key, ""), per_section)
        if not content:
            content = "[résultat absent ou non exécuté]"
        sections.append(f"{key} — {label}:\n{content}")
    return (
        common_prompt(s)
        + "\nTu dois comparer les quatre résultats ci-dessous. Réponds en français avec du texte complet, pas seulement un titre Markdown. "
        + "Commence directement par une phrase de synthèse, puis utilise les rubriques: Synthèse, Comparaison A/B/C/D, Limites, Recommandation. "
        + "Si un résultat est absent, signale-le explicitement sans bloquer la comparaison.\n\n"
        + "\n\n".join(sections)
    )


def ollama_generate_with_retry(prompt: str, s: dict[str, Any], retry_instruction: str,
                               request_log: list[dict[str, Any]] | None = None) -> str:
    response = ollama_generate(prompt, s, request_log)
    if not is_placeholder_llm_response(response):
        return response
    retry_prompt = (
        prompt
        + "\n\nLa réponse précédente était vide ou incomplète (par exemple seulement ###). "
        + retry_instruction
    )
    retry = ollama_generate(retry_prompt, s, request_log)
    return retry if retry.strip() else response


def render_llm_logs(logs: list[dict[str, Any]], label: str) -> None:
    """Show each LLM request and its complete raw response in separate controls."""
    if not logs:
        st.info("Aucun appel effectué pendant cette session.")
        return
    st.caption(
        "Le prompt et tous les paramètres sont affichés; les images base64 sont remplacées "
        "par leur taille et leur SHA-256. La réponse brute est affichée intégralement."
    )
    for index, entry in enumerate(logs, 1):
        st.markdown(f"**Appel {index}**")
        request_details = {key: value for key, value in entry.items() if key != "response_raw"}
        st.json(request_details)
        st.text_area(
            f"Retour brut complet — {label} — appel {index}",
            entry.get("response_raw", "[Aucune réponse HTTP reçue]"),
            height=260,
            key=f"raw_llm_log::{label}::{index}",
        )

def transcribe_video(path: Path, s: dict[str, Any], progress: ProgressCallback | None = None) -> str:
    def run(device, compute):
        if progress: progress(0.1, f"Chargement de Whisper ({device})")
        model=WhisperModel(s["whisper_model_size"], device=device, compute_type=compute)
        if progress: progress(0.25, "Modèle chargé, transcription en cours")
        segs,_=model.transcribe(str(path), language=s["whisper_language"] or None)
        lines=[]; duration = video_metadata(path)["duration_sec"]
        for seg in segs:
            text=re.sub(r"\s+", " ", seg.text).strip()
            if text: lines.append(f"[{fmt_ts(seg.start)} → {fmt_ts(seg.end)}] {text}")
            if progress and duration: progress(0.25 + 0.7 * min(1.0, seg.end / duration), f"Transcription à {fmt_ts(seg.end)}")
        return "\n".join(lines)
    device = "cuda" if s["whisper_device"] == "cuda" else "cpu" if s["whisper_device"] == "cpu" else "auto"
    try: return run(device, s["whisper_compute_type"])
    except Exception as e:
        msg=str(e)
        if s["whisper_fallback_cpu"] and any(x in msg for x in ["libcublas", "libcudnn", "CUDA", "cuda"]):
            return run("cpu", "int8")
        raise TranscriptionError(f"Erreur transcription: {msg}")


def chunks(text: str, n: int) -> list[str]:
    return [text[i:i+n] for i in range(0, len(text), n)] or [""]


def common_prompt(s: dict[str, Any]) -> str:
    return f"Les dialogues peuvent être en {s['dialogue_language']}. L'analyse doit être rédigée en {s['analysis_language']}. Ne réponds pas en russe sauf citation très brève. Objectif: {s['analysis_objective']}."


def default_analysis_prompts(s: dict[str, Any]) -> dict[str, str]:
    """Return the editable instructions used by the three A/B/C analyses."""
    common = common_prompt(s)
    return {
        "analysis_prompt_a": common + "\nDécris ce que l’on comprend uniquement par les images: lieux, personnes, objets, gestes, émotions, textes visibles, limites sans audio/transcript.",
        "analysis_prompt_b": common + "\nRésume en français ce transcript, sujets, intentions, émotions, contexte probable, ambiguïtés nécessitant l’image.",
        "analysis_prompt_b_summary": common + "\nFais une synthèse finale en français de ces résumés.",
        "analysis_prompt_c": common + "\nExplique ce que le transcript permet de comprendre, ce que les images ajoutent, si elles changent l'interprétation ou n'ajoutent presque rien, puis conclus.",
    }


def run_analysis_a(thumbs: list[Thumbnail], s: dict[str, Any], instruction: str,
                   task: ProgressReporter, request_log: list[dict[str, Any]] | None = None) -> str:
    if not thumbs:
        raise OllamaError("Analyse A impossible : extrayez d'abord au moins une vignette.")
    results = []
    batches = chunks_list(thumbs, s["llm_batch_size"])
    for batch_number, part in enumerate(batches, 1):
        task.update((batch_number - 1) / max(1, len(batches)),
                    f"Lot {batch_number}/{len(batches)} envoyé au LLM")
        timestamps = ", ".join(f"#{t.index}={t.timestamp_sec:.2f}s" for t in part)
        results.append(ollama_chat_vision(f"{instruction}\nTimestamps: {timestamps}", part, s,
                                          s["thumbnail_jpeg_quality"], request_log))
    return "\n\n".join(results)


def run_analysis_b(transcript: str, s: dict[str, Any], instruction: str,
                   summary_instruction: str, task: ProgressReporter,
                   request_log: list[dict[str, Any]] | None = None) -> str:
    if not transcript.strip():
        raise OllamaError("Analyse B impossible : le transcript est vide. Transcrivez la vidéo ou saisissez un transcript.")
    results = []
    transcript_parts = chunks(transcript, int(s["transcript_context_max_chars"]))
    for part_number, part in enumerate(transcript_parts, 1):
        task.update((part_number - 1) / (len(transcript_parts) + 1),
                    f"Partie {part_number}/{len(transcript_parts)} analysée")
        results.append(ollama_chat_text(f"{instruction}\n\nTRANSCRIPT À ANALYSER:\n{part}", s,
                                        request_log))
    if len(results) == 1:
        return results[0]
    task.update(len(results) / (len(results) + 1), "Synthèse des parties")
    return ollama_chat_text(f"{summary_instruction}\n\nRÉSUMÉS À SYNTHÉTISER:\n" + "\n".join(results), s,
                            request_log)


def run_analysis_c(thumbs: list[Thumbnail], transcript: str, s: dict[str, Any],
                   instruction: str, task: ProgressReporter,
                   request_log: list[dict[str, Any]] | None = None) -> str:
    if not thumbs:
        raise OllamaError("Analyse C impossible : extrayez d'abord au moins une vignette.")
    if not transcript.strip():
        raise OllamaError("Analyse C impossible : le transcript est vide. Transcrivez la vidéo ou saisissez un transcript.")
    context = build_reduced_transcript_context(transcript, int(s["transcript_context_max_chars"]))
    results = []
    batches = chunks_list(thumbs, s["llm_batch_size"])
    for batch_number, part in enumerate(batches, 1):
        task.update((batch_number - 1) / max(1, len(batches)),
                    f"Lot {batch_number}/{len(batches)} envoyé au LLM")
        prompt = f"{instruction}\nTranscript réduit:\n{context}"
        results.append(ollama_chat_vision(prompt, part, s, s["thumbnail_jpeg_quality"], request_log))
    return "\n\n".join(results)


def sidebar() -> dict[str, Any]:
    if "settings" not in st.session_state: st.session_state.settings = load_settings()
    s=st.session_state.settings
    st.sidebar.header("Paramètres")
    for k,v in default_settings().items(): s.setdefault(k,v)
    s["ollama_base_url"] = st.sidebar.text_input("Ollama URL", s["ollama_base_url"])
    scan_key = f"ollama_models::{s['ollama_base_url']}"
    if scan_key not in st.session_state:
        try:
            st.session_state[scan_key] = {"models": ollama_list_models(s["ollama_base_url"]), "error": ""}
        except OllamaError as exc:
            st.session_state[scan_key] = {"models": [], "error": str(exc)}
    scan = st.session_state[scan_key]
    if st.sidebar.button("Actualiser les modèles Ollama"):
        try:
            scan = {"models": ollama_list_models(s["ollama_base_url"]), "error": ""}
        except OllamaError as exc:
            scan = {"models": [], "error": str(exc)}
        st.session_state[scan_key] = scan
    models = scan["models"]
    if models:
        choices = models if s["ollama_model"] in models else [s["ollama_model"], *models]
        s["ollama_model"] = st.sidebar.selectbox("Modèle Ollama installé", choices,
                                                   index=choices.index(s["ollama_model"]))
        st.sidebar.caption(f"{len(models)} modèle(s) détecté(s) au démarrage.")
    else:
        s["ollama_model"] = st.sidebar.text_input("Modèle Ollama", s["ollama_model"])
        st.sidebar.warning(scan["error"] or "Aucun modèle installé détecté.")
    with st.sidebar.expander("Ollama — paramètres avancés", expanded=True):
        s["ollama_num_ctx"] = st.number_input("num_ctx", 512, 262144, int(s["ollama_num_ctx"]), 1)
        s["ollama_num_predict"] = st.number_input("num_predict", 1, 32768, int(s["ollama_num_predict"]), 1)
        s["ollama_temperature"] = st.number_input("temperature", 0.0, 2.0, float(s["ollama_temperature"]), 0.1)
        s["ollama_top_p"] = st.number_input("top_p", 0.0, 1.0, float(s["ollama_top_p"]), 0.05)
        s["ollama_num_batch"] = st.number_input("num_batch", 1, 8192, int(s["ollama_num_batch"]), 1)
        st.caption("num_ctx : place disponible pour transcript, timestamps et tokens image. num_predict : longueur maximale. temperature : 0 = déterministe. top_p : diversité. num_batch : mémoire/vitesse.")
    s["video_display_max_side"] = st.sidebar.number_input("Taille vidéo affichée — plus grand côté max", 32, 2000, int(s["video_display_max_side"]), 1)
    c1,c2,c3=st.sidebar.columns(3)
    if c1.button("Sauver les paramètres"): save_settings(s); st.sidebar.success("Sauvé")
    if c2.button("Recharger les paramètres"): st.session_state.settings=load_settings(); st.rerun()
    if c3.button("Réinitialiser les paramètres"): st.session_state.settings=default_settings(); st.rerun()
    st.sidebar.download_button("Exporter JSON", json.dumps(s,ensure_ascii=False,indent=2), "video_llm_lab_settings.json")
    imp=st.sidebar.file_uploader("Importer JSON", type=["json"])
    if imp: st.session_state.settings.update(json.loads(imp.read().decode("utf-8"))); st.rerun()
    return s


def main() -> None:
    st.set_page_config(page_title="Video LLM Lab Ollama", layout="wide")
    s=sidebar(); RUNS_DIR.mkdir(exist_ok=True); PROTOCOL_RUNS_DIR.mkdir(exist_ok=True)
    st.title("Video LLM Lab local avec Ollama")
    # Declared before the tabs so every long-running task reports at the top of the page.
    progress_slot = st.empty()
    st.info("Même si les images envoyées sont très petites, Qwen2.5-VL peut les normaliser en interne vers une taille minimale de traitement. Si l’analyse échoue, réduire les images par appel, la taille du transcript ou le nombre de keyframes, et ajuster num_ctx selon l'erreur.")
    tabs=st.tabs(["1. Chargement vidéo","2. Paramètres","3. Extraction des vignettes","4. Transcript","5. Analyse LLM A/B/C","6. Two-pass keyframes","7. Comparaison","8. Diagnostic Ollama","9. Test protocole LLM"])
    with tabs[0]:
        f=st.file_uploader("Charger une vidéo", type=["mp4","mov","mkv","avi","webm"])
        if f:
            run=RUNS_DIR / Path(f.name).stem; run.mkdir(parents=True, exist_ok=True); vp=run / f.name; vp.write_bytes(f.getbuffer())
            st.session_state.video_path=str(vp); meta=video_metadata(vp); st.session_state.video_meta=meta; st.json(meta)
            scale=min(1.0, s["video_display_max_side"] / max(meta["width"], meta["height"]))
            st.video(str(vp)); st.caption(f"Affichage recommandé sans agrandissement: {int(meta['width']*scale)}×{int(meta['height']*scale)} px")
    with tabs[1]:
        s["thumbnail_largest_side_px"] = st.number_input("Taille des vignettes — plus grand côté envoyé au LLM (px)", min_value=MIN_IMAGE_LARGEST_SIDE_PX, max_value=MAX_THUMBNAIL_LARGEST_SIDE_PX, value=int(s["thumbnail_largest_side_px"]), step=1)
        s["thumbnail_interval_sec"] = st.number_input("Intervalle extraction (s)", 0.1, 3600.0, float(s["thumbnail_interval_sec"]), 0.1)
        s["thumbnail_max_frames"] = st.number_input("Nombre maximum de vignettes", 1, 2000, int(s["thumbnail_max_frames"]), 1)
        s["thumbnail_jpeg_quality"] = st.number_input("Qualité JPEG vignettes", 1, 100, int(s["thumbnail_jpeg_quality"]), 1)
        s["thumbnail_gallery_display_width"] = st.number_input("Largeur galerie", 32, 1200, int(s["thumbnail_gallery_display_width"]), 1)
        s["thumbnail_gallery_max_items"] = st.number_input("Images max affichées", 1, 200, int(s["thumbnail_gallery_max_items"]), 1)
        s["llm_batch_size"] = st.number_input("Images par appel LLM", 1, 64, int(s["llm_batch_size"]), 1)
        s["transcript_context_max_chars"] = st.number_input("Transcript max chars", 500, 100000, int(s["transcript_context_max_chars"]), 100)
        s["analysis_language"] = st.text_input("Langue de l’analyse", s["analysis_language"]); s["dialogue_language"] = st.text_input("Langue des dialogues / transcript", s["dialogue_language"])
        s["analysis_objective"] = st.text_area("Objectif d’analyse", s["analysis_objective"])
        s["whisper_model_size"] = st.text_input("Modèle Whisper", s["whisper_model_size"]); s["whisper_device"] = st.selectbox("Device Whisper", ["auto","cpu","cuda"], index=["auto","cpu","cuda"].index(s["whisper_device"]))
        s["whisper_compute_type"] = st.text_input("Compute type Whisper", s["whisper_compute_type"]); s["whisper_language"] = st.text_input("Langue Whisper", s["whisper_language"]); s["whisper_fallback_cpu"] = st.checkbox("Fallback CPU", bool(s["whisper_fallback_cpu"]))
        s["two_pass_max_keyframes"] = st.number_input("Two-pass keyframes max", 1, 100, int(s["two_pass_max_keyframes"]), 1)
        s["two_pass_high_quality_largest_side_px"] = st.number_input("Two-pass haute qualité — plus grand côté", MIN_IMAGE_LARGEST_SIDE_PX, 4096, int(s["two_pass_high_quality_largest_side_px"]), 1)
        s["two_pass_high_quality_jpeg_quality"] = st.number_input("Two-pass qualité JPEG", 1, 100, int(s["two_pass_high_quality_jpeg_quality"]), 1)
        s["two_pass_context_before_sec"] = st.number_input("Contexte avant (s)", 0.0, 60.0, float(s["two_pass_context_before_sec"]), 0.5)
        s["two_pass_context_after_sec"] = st.number_input("Contexte après (s)", 0.0, 60.0, float(s["two_pass_context_after_sec"]), 0.5)
    with tabs[2]:
        if st.button("Extraire les vignettes") and st.session_state.get("video_path"):
            task = ProgressReporter(progress_slot, "Extraction des vignettes")
            try:
                out=Path(st.session_state.video_path).parent / "thumbs"; shutil.rmtree(out, ignore_errors=True)
                st.session_state.thumbnails=extract_thumbnails(Path(st.session_state.video_path), out, s["thumbnail_interval_sec"], s["thumbnail_max_frames"], s["thumbnail_largest_side_px"], s["thumbnail_jpeg_quality"], task.update)
                task.complete(f"{len(st.session_state.thumbnails)} vignette(s) extraite(s)")
            except Exception as exc:
                task.fail(str(exc)); st.error(str(exc))
        thumbs=st.session_state.get("thumbnails", [])
        st.write(f"{len(thumbs)} vignettes")
        for t in thumbs[:s["thumbnail_gallery_max_items"]]: st.image(str(t.path), width=s["thumbnail_gallery_display_width"], caption=f"#{t.index} {t.timestamp_sec:.2f}s {t.width}×{t.height} {t.jpeg_bytes/1024:.1f} Ko")
    with tabs[3]:
        if st.button("Transcrire avec faster-whisper") and st.session_state.get("video_path"):
            task = ProgressReporter(progress_slot, "Transcription")
            try:
                st.session_state.transcript=transcribe_video(Path(st.session_state.video_path), s, task.update)
                task.complete("Transcript terminé")
            except TranscriptionError as e: task.fail(str(e)); st.error(str(e))
        st.session_state.transcript = st.text_area(
            "Transcript nettoyé (modifiable)", st.session_state.get("transcript", ""), height=400
        )
    with tabs[4]:
        thumbs=st.session_state.get("thumbnails", []); transcript=st.session_state.get("transcript", "")
        for key, value in default_analysis_prompts(s).items():
            st.session_state.setdefault(key, value)
        rows=compute_image_payload_report(thumbs, s["llm_batch_size"])
        with st.expander("Images réellement envoyées au LLM", expanded=True):
            st.write(f"Disponibles: {len(thumbs)}; taille configurée: {s['thumbnail_largest_side_px']} px; images/appel: {s['llm_batch_size']}"); st.dataframe(rows)
            if rows:
                st.write(f"Max dimensions envoyées : {max(r['width'] for r in rows)}x{max(r['height'] for r in rows)}; Plus grand côté max : {max(r['largest_side'] for r in rows)} px; Payload base64 max par batch : {max(sum(x['base64_kb'] for x in rows if x['batch_id']==b) for b in set(r['batch_id'] for r in rows)):.1f} Ko")
        with st.expander("Paramètres Ollama actifs"): st.json(ollama_options(s))
        capability_key = f"ollama_capabilities::{s['ollama_base_url']}::{s['ollama_model']}"
        try:
            if capability_key not in st.session_state:
                st.session_state[capability_key] = ollama_model_capabilities(
                    s["ollama_base_url"], s["ollama_model"]
                )
            capabilities = st.session_state[capability_key]
            st.caption(
                f"Capacités déclarées par /api/show pour **{s['ollama_model']}** : "
                + (", ".join(capabilities) or "aucune")
            )
            if "vision" not in capabilities:
                st.warning("Ce modèle est utilisable pour B, mais pas pour A/C qui envoient des images.")
        except OllamaError as exc:
            st.warning(str(exc))
        st.subheader("Consignes envoyées au LLM")
        st.caption("Ces consignes sont modifiables. Les timestamps, le transcript ou les résumés sont ajoutés automatiquement au moment de chaque appel.")
        prompt_a = st.text_area("Consignes A — Images seules", key="analysis_prompt_a", height=130)
        prompt_b = st.text_area("Consignes B — Chaque partie du transcript", key="analysis_prompt_b", height=130)
        prompt_b_summary = st.text_area("Consignes B — Synthèse finale (si plusieurs parties)", key="analysis_prompt_b_summary", height=130)
        prompt_c = st.text_area("Consignes C — Images + transcript", key="analysis_prompt_c", height=130)

        run_all = st.button("Enchaîner A → B → C", type="primary",
                            help="Exécute successivement les trois analyses avec les consignes affichées ci-dessus.")
        if run_all:
            try:
                require_vision_model(s)
                st.session_state.analysis_log_a = []
                st.session_state.analysis_log_b = []
                st.session_state.analysis_log_c = []
                task = ProgressReporter(progress_slot, "Étape 1/3 — Analyse A")
                st.session_state.analysis_a = run_analysis_a(thumbs, s, prompt_a, task, st.session_state.analysis_log_a)
                task.complete("Étape 1/3 terminée")
                task = ProgressReporter(progress_slot, "Étape 2/3 — Analyse B")
                st.session_state.analysis_b = run_analysis_b(transcript, s, prompt_b, prompt_b_summary, task, st.session_state.analysis_log_b)
                task.complete("Étape 2/3 terminée")
                task = ProgressReporter(progress_slot, "Étape 3/3 — Analyse C")
                st.session_state.analysis_c = run_analysis_c(thumbs, transcript, s, prompt_c, task, st.session_state.analysis_log_c)
                task.complete("Les analyses A, B et C sont terminées")
                st.success("Les trois étapes A → B → C ont été exécutées.")
            except OllamaError as e:
                task.fail(str(e)); st.error(str(e))
        if st.button("A — Images seules"):
            task = ProgressReporter(progress_slot, "Analyse A — Images seules")
            try:
                require_vision_model(s)
                st.session_state.analysis_log_a=[]
                st.session_state.analysis_a=run_analysis_a(thumbs, s, prompt_a, task, st.session_state.analysis_log_a)
                task.complete("Analyse A terminée")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        if st.button("B — Transcript seul"):
            task = ProgressReporter(progress_slot, "Analyse B — Transcript seul")
            try:
                st.session_state.analysis_log_b=[]
                st.session_state.analysis_b=run_analysis_b(transcript, s, prompt_b, prompt_b_summary, task, st.session_state.analysis_log_b)
                task.complete("Analyse B terminée")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        if st.button("C — Images + transcript"):
            task = ProgressReporter(progress_slot, "Analyse C — Images + transcript")
            try:
                require_vision_model(s)
                st.session_state.analysis_log_c=[]
                st.session_state.analysis_c=run_analysis_c(thumbs, transcript, s, prompt_c, task, st.session_state.analysis_log_c)
                task.complete("Analyse C terminée")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        for key,label in [("analysis_a","A"),("analysis_b","B"),("analysis_c","C")]:
            st.text_area(label, st.session_state.get(key,""), height=180)
            with st.expander(f"Log exact des appels LLM — cas {label}"):
                logs = st.session_state.get(f"analysis_log_{label.lower()}", [])
                render_llm_logs(logs, f"cas {label}")
    with tabs[5]:
        thumbs=st.session_state.get("thumbnails", []); transcript=st.session_state.get("transcript", "")
        st.subheader("D1 — Sélectionner les keyframes")
        if st.button("Sélectionner les keyframes"):
            task = ProgressReporter(progress_slot, "Sélection des keyframes")
            try:
                st.session_state.analysis_log_d1 = []
                task.update(0.15, "Envoi des vignettes au LLM")
                prompt=common_prompt(s)+"""\nTu reçois une série de vignettes basse résolution extraites chronologiquement d’une vidéo. Ton rôle n’est pas encore de décrire toute la vidéo. Ton rôle est de choisir les images qui méritent une deuxième analyse en meilleure résolution. Sélectionne les images importantes pour comprendre la personne, le lieu, les objets, gestes, émotions, texte visible, changements de scène, moments où l’image ajoute du contexte au transcript. Retourne uniquement un JSON valide au format {\"selected_keyframes\":[{\"frame_index\":12,\"timestamp_sec\":34.5,\"priority\":\"high\",\"reason\":\"...\",\"suggested_focus\":\"...\"}]}\nTranscript réduit:\n"""+build_reduced_transcript_context(transcript, 3000)
                raw=ollama_chat_vision(prompt, thumbs, s, s["thumbnail_jpeg_quality"], st.session_state.analysis_log_d1); st.session_state.keyframe_raw=raw; st.session_state.keyframe_json=extract_json_from_text(raw) or {}
                task.complete("Keyframes sélectionnées")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        st.text_area("Réponse D1", st.session_state.get("keyframe_raw",""), height=160); st.json(st.session_state.get("keyframe_json",{}))
        with st.expander("Log exact des appels LLM — sélection D1"):
            render_llm_logs(st.session_state.get("analysis_log_d1", []), "sélection D1")
        st.subheader("D2 — Extraire les keyframes en meilleure qualité")
        if st.button("Extraire keyframes HQ") and st.session_state.get("video_path"):
            task = ProgressReporter(progress_slot, "Extraction des keyframes HQ")
            try:
                selected=st.session_state.get("keyframe_json",{}).get("selected_keyframes",[])
                st.session_state.keyframes_hq=extract_keyframes(Path(st.session_state.video_path), selected, Path(st.session_state.video_path).parent/"keyframes", s["two_pass_max_keyframes"], s["two_pass_high_quality_largest_side_px"], s["two_pass_high_quality_jpeg_quality"], s["two_pass_context_before_sec"], s["two_pass_context_after_sec"], task.update)
                task.complete(f"{len(st.session_state.keyframes_hq)} keyframe(s) extraite(s)")
            except Exception as exc:
                task.fail(str(exc)); st.error(str(exc))
        for t in st.session_state.get("keyframes_hq",[]): st.image(str(t.path), width=s["thumbnail_gallery_display_width"], caption=f"#{t.index} {t.timestamp_sec:.2f}s {t.width}×{t.height} JPEG {t.jpeg_bytes/1024:.1f} Ko base64 {len(base64.b64encode(t.path.read_bytes()))/1024:.1f} Ko")
        st.subheader("D3 — Décrire les keyframes haute qualité")
        if st.button("Analyser keyframes HQ"):
            task = ProgressReporter(progress_slot, "Analyse D — Keyframes HQ")
            try:
                st.session_state.analysis_log_d3 = []
                task.update(0.15, "Envoi des keyframes au LLM")
                prompt=common_prompt(s)+"\nAnalyse ces keyframes haute résolution. Pour chaque keyframe: timestamp, ce que l’image montre, ce que l’image ajoute au transcript, changement d’interprétation, détails visuels, texte visible, utilité faible/moyen/fort. Termine par une synthèse.\nRaisons de sélection:\n"+json.dumps(st.session_state.get("keyframe_json",{}), ensure_ascii=False)+"\nTranscript réduit:\n"+build_reduced_transcript_context(transcript, int(s["transcript_context_max_chars"]))
                st.session_state.analysis_d=ollama_chat_vision(prompt, st.session_state.get("keyframes_hq",[]), s, s["two_pass_high_quality_jpeg_quality"], st.session_state.analysis_log_d3)
                task.complete("Analyse D terminée")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        st.text_area("D — Two-pass keyframes", st.session_state.get("analysis_d",""), height=240)
        with st.expander("Log exact des appels LLM — analyse D3"):
            render_llm_logs(st.session_state.get("analysis_log_d3", []), "analyse D3")
    with tabs[6]:
        if st.button("Comparer A/B/C/D"):
            task = ProgressReporter(progress_slot, "Comparaison A/B/C/D")
            try:
                st.session_state.comparison_log = []
                task.update(0.15, "Génération de la synthèse finale")
                analyses={"A": st.session_state.get("analysis_a", ""), "B": st.session_state.get("analysis_b", ""), "C": st.session_state.get("analysis_c", ""), "D": st.session_state.get("analysis_d", "")}
                prompt=build_comparison_prompt(s, analyses)
                st.session_state.comparison=ollama_generate_with_retry(prompt, s, "Réécris une comparaison complète en français, sans commencer par un titre Markdown isolé.", st.session_state.comparison_log)
                if is_placeholder_llm_response(st.session_state.comparison):
                    st.warning("La réponse Ollama semble encore incomplète. Augmente num_predict et/ou réduis Transcript max chars, puis relance la comparaison.")
                task.complete("Comparaison terminée")
            except OllamaError as e: task.fail(str(e)); st.error(str(e))
        st.text_area("Comparaison finale", st.session_state.get("comparison",""), height=400)
        with st.expander("Log exact des appels LLM — comparaison"):
            render_llm_logs(st.session_state.get("comparison_log", []), "comparaison")
    with tabs[7]:
        st.json({"options": ollama_options(s), "base_url": s["ollama_base_url"], "model": s["ollama_model"]})
        if st.button("Test /api/tags"):
            try:
                r=requests.get(f"{s['ollama_base_url'].rstrip()}/api/tags", timeout=30); st.write(r.status_code); st.text(r.text)
            except Exception as e: st.error(str(e))
        if st.button("Test texte simple"):
            try: st.success(ollama_generate("Réponds uniquement OK.", s))
            except OllamaError as e: st.error(str(e))
        if st.button("Test vision avec une image générée"):
            img=np.full((120,420,3),255,np.uint8); cv2.putText(img,"TEST OLLAMA VISION",(10,70),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0),2)
            tmp=Path(tempfile.gettempdir())/"ollama_vision_test.jpg"; cv2.imwrite(str(tmp), img); t=Thumbnail(1,0,tmp,420,120,tmp.stat().st_size)
            try: st.success(ollama_chat_vision("Lis le texte visible dans cette image.", [t], s, 90))
            except OllamaError as e: st.error(str(e))
    with tabs[8]:
        st.subheader("Capturer la réponse brute d’un modèle")
        st.write("Ce test conserve séparément le JSON décodé et la réponse HTTP brute afin d’identifier, pour chaque modèle, les champs ou marqueurs tels que `<think>`.")
        protocol_endpoint = st.radio("API à tester", ["chat", "generate"], horizontal=True,
                                     format_func=lambda value: f"/api/{value}")
        protocol_prompt = st.text_area("Contenu envoyé", "Explique brièvement ton raisonnement puis réponds : combien font 17 × 6 ?", height=140)
        protocol_image = st.file_uploader("Image optionnelle envoyée au modèle", type=["png", "jpg", "jpeg", "webp"], key="protocol_image")
        if st.button("Exécuter et enregistrer le test", type="primary"):
            images = [base64.b64encode(protocol_image.getvalue()).decode("ascii")] if protocol_image else []
            try:
                report = ollama_protocol_test(protocol_endpoint, protocol_prompt, s, images)
                safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s["ollama_model"])
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                report_path = PROTOCOL_RUNS_DIR / f"{stamp}_{safe_model}_{protocol_endpoint}.json"
                report_text = json.dumps(report, ensure_ascii=False, indent=2)
                report_path.write_text(report_text, encoding="utf-8")
                st.session_state.protocol_report = report
                st.session_state.protocol_report_text = report_text
                st.session_state.protocol_report_path = str(report_path)
                if report["http"]["status_code"] < 400:
                    st.success(f"Trace enregistrée dans {report_path}")
                else:
                    st.error(f"Ollama a répondu HTTP {report['http']['status_code']}; la trace d’erreur a été conservée.")
            except OllamaError as exc:
                st.error(str(exc))
        report = st.session_state.get("protocol_report")
        if report:
            st.caption(f"Modèle: {report['ollama']['model']} · Endpoint: /api/{report['ollama']['endpoint']} · Durée: {report['duration_ms']} ms")
            st.text_area("Réponse HTTP brute (aucun marqueur supprimé)", report["response_raw"], height=260)
            st.json(report["response_json"] if report["response_json"] is not None else {"non_json_response": True})
            st.download_button("Télécharger la trace JSON", st.session_state["protocol_report_text"],
                               file_name=Path(st.session_state["protocol_report_path"]).name,
                               mime="application/json")


def chunks_list(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i:i+n] for i in range(0, len(items), max(1,n))]


if __name__ == "__main__":
    main()
