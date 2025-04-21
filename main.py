# python -m streamlit run main.py

# pip install opencv-python

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import subprocess
import os
import cv2
from yt_dlp import YoutubeDL
from opticalflow import compute_optical_flow_metrics, _get_frame_at_time

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------
def convertir_en_min_sec(seconds: float) -> str:
    """Convertit des secondes en format mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def telecharger_video_et_extraire_audio(video_url: str, rep="downloads"):
    """Télécharge une vidéo YouTube et extrait le WAV mono 16 kHz."""
    os.makedirs(rep, exist_ok=True)
    opts = {"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
            "outtmpl": os.path.join(rep, "%(id)s.%(ext)s"),
            "quiet": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
    vid_id = info["id"]
    video_path = os.path.join(rep, f"{vid_id}.mp4")
    wav_path = os.path.join(rep, f"{vid_id}.wav")
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return video_path, wav_path

def downsample_by_second(data: np.ndarray, times: np.ndarray, samplerate: int):
    """Regroupe le signal audio en intervalles de 1 s et calcule l'enveloppe (min/max)"""
    n = len(data)
    interval = samplerate
    nb = n // interval
    t_int, mn, mx, env = [], [], [], []
    for i in range(nb):
        seg = data[i*interval:(i+1)*interval]
        t_int.append(times[i*interval:(i+1)*interval].mean())
        mn.append(float(seg.min())); mx.append(float(seg.max()))
        env.append(float((seg.min()+seg.max())/2))
    return np.array(t_int), np.array(mn), np.array(mx), np.array(env)

def transcrire_audio_whisper(wav_path: str):
    """Transcrit l'audio avec Whisper (modèle 'small', langue FR)."""
    import whisper
    model = whisper.load_model("small")
    result = model.transcribe(wav_path, language="fr")
    return result.get("segments", [])

def faire_carte_flux(flow_map: np.ndarray) -> np.ndarray:
    """Génère une heatmap JET à partir d'une carte de flux optique."""
    if flow_map.ndim == 3:
        mag = np.linalg.norm(flow_map, axis=2)
    else:
        mag = flow_map
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------

st.title("Analyse amplitude sonore & flux optique synchronisé")
video_url = st.text_input("URL YouTube")
k_value = st.slider("k (intervalle [μ ± kσ])", 1.0, 5.0, 2.0, 0.1)

if st.button("Lancer l’analyse multimodale"):
    if not video_url:
        st.error("Veuillez renseigner une URL YouTube.")
        st.stop()

    # Étape 1 : téléchargement + extraction audio
    st.info("Téléchargement vidéo + extraction audio…")
    try:
        video_path, audio_path = telecharger_video_et_extraire_audio(video_url)
    except Exception as e:
        st.error(f"Erreur téléchargement : {e}")
        st.stop()

    # Étape 2 : lecture du fichier WAV
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data); dur = n/sr
    st.write(f"Durée : {dur:.1f}s — {n} échantillons à {sr} Hz")

    # Étape 3 : binning audio 1s
    times = np.linspace(0, dur, n)
    t_int, mn, mx, env = downsample_by_second(data, times, sr)
    st.info(f"Signal audio divisé en {len(t_int)} intervalles de 1 s")

    # Étape 4 : seuils audio
    mu = env.mean(); sigma = env.std()
    mu = 0.0 if abs(mu)<1e-6 else mu
    sigma = 0.0 if abs(sigma)<1e-6 else sigma
    lb, ub = mu - k_value*sigma, mu + k_value*sigma
    st.write(f"Amplitude audio – μ={mu:.4f}, σ={sigma:.4f}")
    st.write(f"Seuils : [{lb:.4f}, {ub:.4f}]")

    # Étape 5 : détection anomalies audio
    idx = np.where((env<lb)|(env>ub))[0]
    t_out, env_out = t_int[idx], env[idx]
    st.info(f"{len(idx)} anomalies audio détectées")

    # Étape 6 : graphique amplitude
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_int, t_int[::-1]]),
        y=np.concatenate([mn, mx[::-1]]),
        fill='toself', fillcolor='rgba(255,255,0,0.2)', line=dict(width=0), name='Enveloppe'
    ))
    fig.add_trace(go.Scatter(x=t_int, y=env, mode='lines', name='Env moyenne'))
    fig.add_trace(go.Scatter(x=t_out, y=env_out, mode='markers', marker=dict(color='red', size=8), name='Anomalies'))
    fig.update_layout(xaxis_title='Temps (s)', yaxis_title='Amplitude')
    st.plotly_chart(fig, use_container_width=True)

    # Étape 7 : transcription audio
    st.info("Transcription audio avec Whisper…")
    segments = transcrire_audio_whisper(audio_path)

    # Notice d'interprétation
    st.markdown("""
**Notice d'interprétation**
- **Auteur de la méthode :** Gunnar Farneback
- **Principe :** extraction de milliers de vecteurs de déplacement (un par pixel) par paire de frames
- **Calcul :** moyenne de la norme de ces vecteurs → score unique
- **Signification :** score élevé = mouvement rapide, score bas = pause
- **Couleurs JET :** bleu (faible) → vert → jaune → rouge (très fort)
""")

    # Étape 8 : analyse du flux optique
    st.subheader("Flux optique sur anomalies")
    flows = compute_optical_flow_metrics(video_path, t_out.tolist(), dt=1.0)
    for i, evt in enumerate(flows):
        t = evt['time']
        amplitude_audio = env_out[i]
        mag_prev = evt['mag_prev']; mag_next = evt['mag_next']
        st.markdown(f"**T = {t:.0f}s**  •  Amplitude audio = {amplitude_audio:.2f}  •  mag_t-1 = {mag_prev:.2f}  •  mag_t = {mag_next:.2f}")

        # 3 images brutes (t-1, t, t+1)
        c1, c2, c3 = st.columns(3)
        c1.image(evt['frame_prev'], channels='BGR', caption=f'Image brute t-1={t-1:.0f}s')
        c2.image(evt['frame'],      channels='BGR', caption=f'Image brute t={t:.0f}s')
        c3.image(evt['frame_next'], channels='BGR', caption=f'Image brute t+1={t+1:.0f}s')

        # 2 heatmaps (flux t-1→t, flux t→t+1)
        h1, h2 = st.columns(2)
        heatmap_prev = faire_carte_flux(evt['flow_map_prev'])
        heatmap_next = faire_carte_flux(evt['flow_map_next'])
        h1.image(heatmap_prev, channels='BGR', caption='Flux optique t-1→t')
        h2.image(heatmap_next, channels='BGR', caption='Flux optique t→t+1')

        # Superposition calorimétrique
        sb1, sb2 = st.columns(2)
        overlay_prev = cv2.addWeighted(evt['frame_prev'], 0.7, heatmap_prev, 0.3, 0)
        overlay_next = cv2.addWeighted(evt['frame'],      0.7, heatmap_next, 0.3, 0)
        sb1.image(overlay_prev, channels='BGR', caption='Superposition t-1→t')
        sb2.image(overlay_next, channels='BGR', caption='Superposition t→t+1')

        # Transcriptions t-1, t et t+1
        st.subheader("Transcriptions aux instants t-1, t et t+1")
        for offset, label in [(-1,'t-1'), (0,'t'), (1,'t+1')]:
            tp = t + offset
            segs = [s['text'].strip() for s in segments if s['start'] <= tp <= s['end']]
            uniques = list(dict.fromkeys(segs))
            st.write(f"**Segments {label} ({tp:.0f}s)**")
            if uniques:
                for txt in uniques:
                    st.write(f"- {txt}")
            else:
                st.write("_aucun segment_")

    st.success("Analyse terminée !")
