############
# Analyse de l'amplitude sonore and opticalflow
# Stéphane Meurisse
# www.codeandcortex.fr
# Date : 24-04-2024
############

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import subprocess
import os
import cv2
import whisper
from yt_dlp import YoutubeDL
from opticalflow import compute_optical_flow_metrics, _get_frame_at_time
import imageio_ffmpeg

# ----------------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------------

def convertir_en_min_sec(seconds: float) -> str:
    """Convertit des secondes en mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def telecharger_video_et_extraire_audio(video_url: str, rep="downloads"):
    """
    Télécharge la vidéo YouTube et en extrait le WAV (16 kHz mono)
    en forçant yt_dlp et ffmpeg à utiliser le binaire embarqué par imageio-ffmpeg.
    """
    os.makedirs(rep, exist_ok=True)

    # 1) Chemin vers le binaire ffmpeg fourni par imageio-ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    # 2) Options pour yt_dlp : on pointe ffmpeg vers notre binaire Python
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": os.path.join(rep, "%(id)s.%(ext)s"),
        "quiet": True,
        "ffmpeg_location": ffmpeg_bin,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    vid_id     = info["id"]
    video_path = os.path.join(rep, f"{vid_id}.mp4")
    wav_path   = os.path.join(rep, f"{vid_id}.wav")

    # 3) Extraction audio avec le même binaire
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          check=True)

    return video_path, wav_path


def downsample_by_second(data: np.ndarray, times: np.ndarray, samplerate: int):
    """Binning audio en intervalles de 1 s et calcul enveloppe min/max."""
    n = len(data)
    step = samplerate
    nb = n // step
    t_int, mn, mx, env = [], [], [], []
    for i in range(nb):
        seg = data[i*step:(i+1)*step]
        t_int.append(times[i*step:(i+1)*step].mean())
        mn.append(float(seg.min())); mx.append(float(seg.max()))
        env.append(float((seg.min()+seg.max())/2))
    return np.array(t_int), np.array(mn), np.array(mx), np.array(env)


def chercher_pic(data: np.ndarray, sr: int, t_center: float) -> float:
    """Retourne timestamp du pic absolu d’amplitude autour de t_center ±0.5s."""
    demi = sr // 2
    idx = int(t_center * sr)
    start, end = max(idx-demi,0), min(idx+demi,len(data))
    window = data[start:end]
    if window.size == 0:
        return t_center
    rel = np.argmax(np.abs(window))
    return (start+rel)/sr


def transcrire_audio_whisper(wav_path: str) -> list[dict]:
    """
    Transcrit l'audio en français via Whisper sans faire appel à ffmpeg :
    - on lit le WAV avec soundfile (déjà à 16 kHz mono),
    - on passe le ndarray directement à model.transcribe.
    """
    # 1) Charger le WAV avec soundfile
    data, sr = sf.read(wav_path)
    # Whisper attend du 16 kHz. Si ce n'est pas le cas, il faut resampler.
    if sr != 16000:
        import numpy as np
        from scipy.signal import resample
        num = int(len(data) * 16000 / sr)
        data = resample(data, num)
        sr = 16000

    # 2) Charger le modèle et transcrire à partir du tableau
    model = whisper.load_model("small")
    # On désactive l'utilisation du binaire externe en passant l'array
    result = model.transcribe(
        data,            # tableau numpy (et non chemin de fichier)
        fp16=False,      # selon votre GPU/CPU
        language="fr"    # force le français
    )
    return result.get("segments", [])


def faire_carte_flux(flow_map: np.ndarray) -> np.ndarray:
    """Génère heatmap JET d'une carte de flux optique."""
    mag = np.linalg.norm(flow_map,axis=2) if flow_map.ndim==3 else flow_map
    norm = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8),cv2.COLORMAP_JET)


def superposer_vecteurs(frame: np.ndarray, flow_map: np.ndarray, step: int=16) -> np.ndarray:
    """Superpose flèches de flux sur une frame vidéo."""
    img = frame.copy(); h,w = img.shape[:2]
    fx,fy = flow_map[...,0],flow_map[...,1]
    for y in range(0,h,step):
        for x in range(0,w,step):
            dx,dy = int(fx[y,x]),int(fy[y,x])
            cv2.arrowedLine(img,(x,y),(x+dx,y+dy),(0,255,0),1,tipLength=0.3)
    return img

# ----------------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------------

st.title("Analyse amplitude sonore & flux optique synchronisé")
video_url = st.text_input("URL YouTube")
k_value   = st.slider("k (intervalle [μ ± kσ])",1.0,5.0,2.0,0.1)

if st.button("Lancer l’analyse"):
    if not video_url:
        st.error("Veuillez renseigner une URL YouTube.")
        st.stop()

    # 1) Téléchargement + vidéo
    st.info("Téléchargement vidéo et extraction audio…")
    video_path,audio_path = telecharger_video_et_extraire_audio(video_url)
    st.video(video_path)

    # 2) Lecture audio
    data,sr = sf.read(audio_path)
    if data.ndim>1: data = data.mean(axis=1)
    dur = len(data)/sr
    st.write(f"Durée : {dur:.1f}s — {len(data)} échantillons à {sr}Hz")

    # 3) Intervalle audio
    times = np.linspace(0,dur,len(data))
    t_int,mn,mx,env = downsample_by_second(data,times,sr)

    # 4) Transcription segments Whisper
    st.info("Transcription audio avec Whisper…")
    segments = transcrire_audio_whisper(audio_path)

    # 5) Seuils audio & anomalies
    mu,sigma = env.mean(),env.std()
    mu =0 if abs(mu)<1e-6 else mu; sigma=0 if abs(sigma)<1e-6 else sigma
    lb,ub = mu-k_value*sigma, mu+k_value*sigma
    idx = np.where((env<lb)|(env>ub))[0]
    t_out,env_out = t_int[idx],env[idx]
    st.info(f"{len(idx)} observations atypiques détectées")

    # 6) Graphique amplitude enveloppe blanche
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t_int,t_int[::-1]],y=np.r_[mn,mx[::-1]],fill='toself',fillcolor='rgba(255,255,255,0.2)',line=dict(width=0),name='Enveloppe'))
    fig.add_trace(go.Scatter(x=t_int,y=env,mode='lines',name='Enveloppe moyenne'))
    fig.add_trace(go.Scatter(x=t_out,y=env_out,mode='markers',marker=dict(color='red',size=8),name='Anomalies'))
    fig.update_layout(xaxis_title='Temps (s)',yaxis_title='Amplitude audio')
    st.plotly_chart(fig,use_container_width=True)

    # 4) Explications et interprétations
    st.subheader("Interprétation des résultats")
    st.markdown(
        "**Qu'est-ce que la magnitude optique ?**"
        "La **magnitude optique** correspond à la moyenne des normes des vecteurs de déplacement calculés "
        "entre deux images consécutives par l'algorithme Farneback. Elle quantifie l'intensité du mouvement visuel :"
    )
    st.markdown("- **Valeurs élevées** : mouvements rapides ou importants")
    st.markdown("- **Valeurs faibles** : mouvements lents ou quasi-statiques")

    st.markdown("**Calcul de l'observation atypique audio :**")
    st.markdown("Une observation audio atypique est détectée lorsque l'amplitude moyenne de l'enveloppe audio dépasse le seuil défini par μ ± kσ,")
    st.markdown("où μ est la moyenne des amplitudes sur la vidéo et σ leur écart-type. Cela permet d'identifier des pics sonores significatifs.")

    st.markdown(
        "**Flux optique :**"
        "Le flux optique (Farneback) mesure les déplacements de pixels entre deux images consécutives."
        "Une heatmap JET traduit ces déplacements en intensité de mouvement, du bleu (faible) au rouge (fort)."
    )
    st.markdown(
        "**Superposition :**"
        "La superposition de la heatmap sur l'image d'origine met en évidence les zones de mouvement significatif,"
        "conservant la perception visuelle du contenu tout en signalant le mouvement."
    )
    st.markdown(
        "**Vecteurs de flux :**"
        "Les flèches tracées représentent les vecteurs de déplacement (dx, dy) de blocs de pixels."
        "Leur densité et leur orientation illustrent la direction et l'amplitude du mouvement."
    )

    # 7) Analyse de chaque anomalie (t-1, t, t+1)
    offsets = [-1, 0, 1]
    for i, t0 in enumerate(t_out):
        st.subheader(f"Observation #{i + 1} — anomalie à {t0:.1f}s")

        # a) Pic exact et amplitude audio
        t_pic = chercher_pic(data, sr, t0)
        amp = env_out[i]
        st.markdown(f"**Pic amplitude :** {t_pic:.2f}s  •  **Amplitude audio :** {amp:.2f}")

        # b) Calcul du flux optique et récupération des magnitudes
        flows = compute_optical_flow_metrics(video_path, [t_pic], dt=1.0)
        evt = flows[0]
        mag_prev = evt['mag_prev']
        mag_next = evt['mag_next']
        st.markdown(f"**mag_t-1** : {mag_prev:.2f}  •  **mag_t** : {mag_next:.2f}")

        # c) Affichage des images brutes
        st.markdown("**Images brutes**")
        cols = st.columns(3)
        cap = cv2.VideoCapture(video_path)
        for col, off in zip(cols, offsets):
            img = _get_frame_at_time(cap, t_pic + off)
            col.image(img, channels='BGR', caption=f"t_pic+{off}s")
        cap.release()

        # d) Heatmaps du flux optique
        st.markdown("**Heatmaps (flux optique)**")
        h1, h2 = st.columns(2)
        heat_prev = faire_carte_flux(evt['flow_map_prev'])
        heat_next = faire_carte_flux(evt['flow_map_next'])
        h1.image(heat_prev, channels='BGR', caption="Flux t-1→t")
        h2.image(heat_next, channels='BGR', caption="Flux t→t+1")

        # e) Superposition heatmap + image
        st.markdown("**Superposition heatmap + image**")
        s1, s2 = st.columns(2)
        sup_prev = cv2.addWeighted(evt['frame_prev'], 0.7, heat_prev, 0.3, 0)
        sup_next = cv2.addWeighted(evt['frame'], 0.7, heat_next, 0.3, 0)
        s1.image(sup_prev, channels='BGR', caption="Superp. t-1→t")
        s2.image(sup_next, channels='BGR', caption="Superp. t→t+1")

        # f) Vecteurs de flux
        st.markdown("**Vecteurs de flux**")
        v1, v2 = st.columns(2)
        vec_prev = superposer_vecteurs(evt['frame_prev'], evt['flow_map_prev'])
        vec_next = superposer_vecteurs(evt['frame'], evt['flow_map_next'])
        v1.image(vec_prev, channels='BGR', caption="Vecteurs t-1→t")
        v2.image(vec_next, channels='BGR', caption="Vecteurs t→t+1")

        # g) Segments de texte autour du pic
        st.subheader("Segments transcrits autour du pic audio")
        for off in offsets:
            tp = t_pic + off
            seg = next((s for s in segments if s['start'] <= tp <= s['end']), None)
            texte = seg['text'].strip() if seg else "_aucun segment_"
            st.markdown(f"**t+{off}s** : {texte}")


st.success("Analyse terminée !")
