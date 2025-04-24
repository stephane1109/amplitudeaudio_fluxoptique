import cv2
import numpy as np

def _get_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
    """
    Se positionne à time_sec (s) et renvoie la frame correspondante.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Impossible de lire la frame à {time_sec}s")
    return frame

def compute_optical_flow_metrics(video_path: str, event_times: list[float], dt: float = 1.0) -> list[dict]:
    """
    Pour chaque t dans event_times, extrait les frames à t-dt, t, t+dt,
    calcule le flux Farneback entre (t-dt)->t et t->(t+dt),
    renvoie pour chaque événement :
      - time, mag_prev, mag_next
      - frame_prev, frame, frame_next
      - flow_map_prev, flow_map_next (cartes de flux)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d’ouvrir la vidéo : {video_path}")

    results = []
    for t in event_times:
        t1 = max(t - dt, 0)
        t2 = t
        t3 = t + dt

        f1 = _get_frame_at_time(cap, t1)
        f2 = _get_frame_at_time(cap, t2)
        f3 = _get_frame_at_time(cap, t3)

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        g3 = cv2.cvtColor(f3, cv2.COLOR_BGR2GRAY)

        flow_prev = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_next = cv2.calcOpticalFlowFarneback(
            g2, g3, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag_prev = float(np.mean(np.linalg.norm(flow_prev, axis=2)))
        mag_next = float(np.mean(np.linalg.norm(flow_next, axis=2)))

        results.append({
            "time":        t,
            "mag_prev":    mag_prev,
            "mag_next":    mag_next,
            "frame_prev":  f1,
            "frame":       f2,
            "frame_next":  f3,
            "flow_map_prev": flow_prev,
            "flow_map_next": flow_next
        })

    cap.release()
    return results

