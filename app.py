from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import streamlit as st
from faster_whisper import WhisperModel

MIN_IMAGE_LARGEST_SIDE_PX = 32
DEFAULT_THUMBNAIL_LARGEST_SIDE_PX = 400
MAX_THUMBNAIL_LARGEST_SIDE_PX = 1600
RUNS_DIR = Path("video_llm_lab_runs")
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


def extract_thumbnails(video_path: Path, out_dir: Path, interval: float, max_frames: int, largest_side: int, quality: int) -> list[Thumbnail]:
    cap = cv2.VideoCapture(str(video_path)); meta = video_metadata(video_path)
    thumbs = []
    for i in range(max_frames):
        ts = i * interval
        if meta["duration_sec"] and ts > meta["duration_sec"]: break
        frame = extract_frame_at(cap, ts)
        if frame is None: break
        p = out_dir / f"thumb_{i+1:06d}_{ts:.2f}s.jpg"
        t = save_jpeg(frame, p, quality, largest_side); t.index = i + 1; t.timestamp_sec = ts; thumbs.append(t)
    cap.release(); return thumbs


def extract_keyframes(video_path: Path, selected: list[dict[str, Any]], out_dir: Path, max_items: int, largest_side: int, quality: int, before: float, after: float) -> list[Thumbnail]:
    cap = cv2.VideoCapture(str(video_path)); out=[]; seen=set(); idx=1
    for item in selected[:max_items]:
        ts0 = float(item.get("timestamp_sec", 0))
        for ts in [ts0 - before, ts0, ts0 + after]:
            ts = max(0.0, ts)
            key = round(ts, 2)
            if key in seen: continue
            seen.add(key); frame = extract_frame_at(cap, ts)
            if frame is None: continue
            p = out_dir / f"keyframe_{idx:06d}_{ts:.2f}s.jpg"
            t = save_jpeg(frame, p, quality, largest_side); t.index=idx; t.timestamp_sec=ts; out.append(t); idx += 1
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


def ollama_options(s: dict[str, Any]) -> dict[str, Any]:
    return {"num_ctx": s["ollama_num_ctx"], "num_predict": s["ollama_num_predict"], "temperature": s["ollama_temperature"], "top_p": s["ollama_top_p"], "num_batch": s["ollama_num_batch"]}


def ollama_generate(prompt: str, s: dict[str, Any]) -> str:
    r = requests.post(f"{s['ollama_base_url'].rstrip('/')}/api/generate", json={"model": s["ollama_model"], "prompt": prompt, "stream": False, "options": ollama_options(s)}, timeout=600)
    if not r.ok: raise_ollama_error(r, "Échec Ollama /api/generate")
    return r.json().get("response", "")


def ollama_chat_vision(prompt: str, thumbs: list[Thumbnail], s: dict[str, Any], quality: int) -> str:
    images=[thumbnail_to_ollama_base64(t, quality)[0] for t in thumbs]
    payload={"model": s["ollama_model"], "messages":[{"role":"user","content":prompt,"images":images}], "stream":False, "options":ollama_options(s)}
    r=requests.post(f"{s['ollama_base_url'].rstrip('/')}/api/chat", json=payload, timeout=900)
    if r.ok: return r.json().get("message",{}).get("content","")
    if len(thumbs) > 1:
        parts=[]
        for t in thumbs:
            parts.append(ollama_chat_vision(prompt + f"\nImage unique: index {t.index}, timestamp {t.timestamp_sec:.2f}s", [t], s, quality))
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


def ollama_generate_with_retry(prompt: str, s: dict[str, Any], retry_instruction: str) -> str:
    response = ollama_generate(prompt, s)
    if not is_placeholder_llm_response(response):
        return response
    retry_prompt = (
        prompt
        + "\n\nLa réponse précédente était vide ou incomplète (par exemple seulement ###). "
        + retry_instruction
    )
    retry = ollama_generate(retry_prompt, s)
    return retry if retry.strip() else response

def transcribe_video(path: Path, s: dict[str, Any]) -> str:
    def run(device, compute):
        model=WhisperModel(s["whisper_model_size"], device=device, compute_type=compute)
        segs,_=model.transcribe(str(path), language=s["whisper_language"] or None)
        lines=[]
        for seg in segs:
            text=re.sub(r"\s+", " ", seg.text).strip()
            if text: lines.append(f"[{fmt_ts(seg.start)} → {fmt_ts(seg.end)}] {text}")
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


def sidebar() -> dict[str, Any]:
    if "settings" not in st.session_state: st.session_state.settings = load_settings()
    s=st.session_state.settings
    st.sidebar.header("Paramètres")
    for k,v in default_settings().items(): s.setdefault(k,v)
    s["ollama_base_url"] = st.sidebar.text_input("Ollama URL", s["ollama_base_url"])
    s["ollama_model"] = st.sidebar.text_input("Modèle Ollama", s["ollama_model"])
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
    s=sidebar(); RUNS_DIR.mkdir(exist_ok=True)
    st.title("Video LLM Lab local avec Ollama")
    st.info("Même si les images envoyées sont très petites, Qwen2.5-VL peut les normaliser en interne vers une taille minimale de traitement. Si l’analyse échoue, réduire les images par appel, la taille du transcript ou le nombre de keyframes, et ajuster num_ctx selon l'erreur.")
    tabs=st.tabs(["1. Chargement vidéo","2. Paramètres","3. Extraction des vignettes","4. Transcript","5. Analyse LLM A/B/C","6. Two-pass keyframes","7. Comparaison","8. Diagnostic Ollama"])
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
            out=Path(st.session_state.video_path).parent / "thumbs"; shutil.rmtree(out, ignore_errors=True)
            st.session_state.thumbnails=extract_thumbnails(Path(st.session_state.video_path), out, s["thumbnail_interval_sec"], s["thumbnail_max_frames"], s["thumbnail_largest_side_px"], s["thumbnail_jpeg_quality"])
        thumbs=st.session_state.get("thumbnails", [])
        st.write(f"{len(thumbs)} vignettes")
        for t in thumbs[:s["thumbnail_gallery_max_items"]]: st.image(str(t.path), width=s["thumbnail_gallery_display_width"], caption=f"#{t.index} {t.timestamp_sec:.2f}s {t.width}×{t.height} {t.jpeg_bytes/1024:.1f} Ko")
    with tabs[3]:
        if st.button("Transcrire avec faster-whisper") and st.session_state.get("video_path"):
            try: st.session_state.transcript=transcribe_video(Path(st.session_state.video_path), s)
            except TranscriptionError as e: st.error(str(e))
        st.text_area("Transcript nettoyé", st.session_state.get("transcript",""), height=400)
    with tabs[4]:
        thumbs=st.session_state.get("thumbnails", []); transcript=st.session_state.get("transcript", "")
        rows=compute_image_payload_report(thumbs, s["llm_batch_size"])
        with st.expander("Images réellement envoyées au LLM", expanded=True):
            st.write(f"Disponibles: {len(thumbs)}; taille configurée: {s['thumbnail_largest_side_px']} px; images/appel: {s['llm_batch_size']}"); st.dataframe(rows)
            if rows:
                st.write(f"Max dimensions envoyées : {max(r['width'] for r in rows)}x{max(r['height'] for r in rows)}; Plus grand côté max : {max(r['largest_side'] for r in rows)} px; Payload base64 max par batch : {max(sum(x['base64_kb'] for x in rows if x['batch_id']==b) for b in set(r['batch_id'] for r in rows)):.1f} Ko")
        with st.expander("Paramètres Ollama actifs"): st.json(ollama_options(s))
        if st.button("A — Images seules"):
            try:
                res=[]
                for part in chunks_list(thumbs, s["llm_batch_size"]):
                    prompt=common_prompt(s)+"\nDécris ce que l’on comprend uniquement par les images: lieux, personnes, objets, gestes, émotions, textes visibles, limites sans audio/transcript. Timestamps: "+", ".join(f"#{t.index}={t.timestamp_sec:.2f}s" for t in part)
                    res.append(ollama_chat_vision(prompt, part, s, s["thumbnail_jpeg_quality"]))
                st.session_state.analysis_a="\n\n".join(res)
            except OllamaError as e: st.error(str(e))
        if st.button("B — Transcript seul"):
            try:
                parts=[]
                for c in chunks(transcript, int(s["transcript_context_max_chars"])):
                    parts.append(ollama_generate(common_prompt(s)+"\nRésume en français ce transcript, sujets, intentions, émotions, contexte probable, ambiguïtés nécessitant l’image.\n"+c, s))
                st.session_state.analysis_b=ollama_generate(common_prompt(s)+"\nFais une synthèse finale en français de ces résumés:\n"+"\n".join(parts), s) if len(parts)>1 else parts[0]
            except OllamaError as e: st.error(str(e))
        if st.button("C — Images + transcript"):
            try:
                ctx=build_reduced_transcript_context(transcript, int(s["transcript_context_max_chars"])); res=[]
                for part in chunks_list(thumbs, s["llm_batch_size"]):
                    prompt=common_prompt(s)+"\nExplique ce que le transcript permet de comprendre, ce que les images ajoutent, si elles changent l'interprétation ou n'ajoutent presque rien, puis conclus.\nTranscript réduit:\n"+ctx
                    res.append(ollama_chat_vision(prompt, part, s, s["thumbnail_jpeg_quality"]))
                st.session_state.analysis_c="\n\n".join(res)
            except OllamaError as e: st.error(str(e))
        for key,label in [("analysis_a","A"),("analysis_b","B"),("analysis_c","C")]: st.text_area(label, st.session_state.get(key,""), height=180)
    with tabs[5]:
        thumbs=st.session_state.get("thumbnails", []); transcript=st.session_state.get("transcript", "")
        st.subheader("D1 — Sélectionner les keyframes")
        if st.button("Sélectionner les keyframes"):
            try:
                prompt=common_prompt(s)+"""\nTu reçois une série de vignettes basse résolution extraites chronologiquement d’une vidéo. Ton rôle n’est pas encore de décrire toute la vidéo. Ton rôle est de choisir les images qui méritent une deuxième analyse en meilleure résolution. Sélectionne les images importantes pour comprendre la personne, le lieu, les objets, gestes, émotions, texte visible, changements de scène, moments où l’image ajoute du contexte au transcript. Retourne uniquement un JSON valide au format {\"selected_keyframes\":[{\"frame_index\":12,\"timestamp_sec\":34.5,\"priority\":\"high\",\"reason\":\"...\",\"suggested_focus\":\"...\"}]}\nTranscript réduit:\n"""+build_reduced_transcript_context(transcript, 3000)
                raw=ollama_chat_vision(prompt, thumbs, s, s["thumbnail_jpeg_quality"]); st.session_state.keyframe_raw=raw; st.session_state.keyframe_json=extract_json_from_text(raw) or {}
            except OllamaError as e: st.error(str(e))
        st.text_area("Réponse D1", st.session_state.get("keyframe_raw",""), height=160); st.json(st.session_state.get("keyframe_json",{}))
        st.subheader("D2 — Extraire les keyframes en meilleure qualité")
        if st.button("Extraire keyframes HQ") and st.session_state.get("video_path"):
            selected=st.session_state.get("keyframe_json",{}).get("selected_keyframes",[])
            st.session_state.keyframes_hq=extract_keyframes(Path(st.session_state.video_path), selected, Path(st.session_state.video_path).parent/"keyframes", s["two_pass_max_keyframes"], s["two_pass_high_quality_largest_side_px"], s["two_pass_high_quality_jpeg_quality"], s["two_pass_context_before_sec"], s["two_pass_context_after_sec"])
        for t in st.session_state.get("keyframes_hq",[]): st.image(str(t.path), width=s["thumbnail_gallery_display_width"], caption=f"#{t.index} {t.timestamp_sec:.2f}s {t.width}×{t.height} JPEG {t.jpeg_bytes/1024:.1f} Ko base64 {len(base64.b64encode(t.path.read_bytes()))/1024:.1f} Ko")
        st.subheader("D3 — Décrire les keyframes haute qualité")
        if st.button("Analyser keyframes HQ"):
            try:
                prompt=common_prompt(s)+"\nAnalyse ces keyframes haute résolution. Pour chaque keyframe: timestamp, ce que l’image montre, ce que l’image ajoute au transcript, changement d’interprétation, détails visuels, texte visible, utilité faible/moyen/fort. Termine par une synthèse.\nRaisons de sélection:\n"+json.dumps(st.session_state.get("keyframe_json",{}), ensure_ascii=False)+"\nTranscript réduit:\n"+build_reduced_transcript_context(transcript, int(s["transcript_context_max_chars"]))
                st.session_state.analysis_d=ollama_chat_vision(prompt, st.session_state.get("keyframes_hq",[]), s, s["two_pass_high_quality_jpeg_quality"])
            except OllamaError as e: st.error(str(e))
        st.text_area("D — Two-pass keyframes", st.session_state.get("analysis_d",""), height=240)
    with tabs[6]:
        if st.button("Comparer A/B/C/D"):
            try:
                analyses={"A": st.session_state.get("analysis_a", ""), "B": st.session_state.get("analysis_b", ""), "C": st.session_state.get("analysis_c", ""), "D": st.session_state.get("analysis_d", "")}
                prompt=build_comparison_prompt(s, analyses)
                st.session_state.comparison=ollama_generate_with_retry(prompt, s, "Réécris une comparaison complète en français, sans commencer par un titre Markdown isolé.")
                if is_placeholder_llm_response(st.session_state.comparison):
                    st.warning("La réponse Ollama semble encore incomplète. Augmente num_predict et/ou réduis Transcript max chars, puis relance la comparaison.")
            except OllamaError as e: st.error(str(e))
        st.text_area("Comparaison finale", st.session_state.get("comparison",""), height=400)
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


def chunks_list(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i:i+n] for i in range(0, len(items), max(1,n))]


if __name__ == "__main__":
    main()
