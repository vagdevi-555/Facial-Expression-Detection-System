import cv2
import time
from collections import deque
from deepface import DeepFace

# ================= CONFIG =================
CAMERA_INDEX = 0
WINDOW_NAME = "Facial Expression Intelligence System"

ANALYZE_EVERY_N_FRAMES = 15     # controls analysis frequency
SMOOTHING_FRAMES = 15           # temporal smoothing (~1.5 sec)
EMOTION_HOLD_SECONDS = 2.0      # commit delay (final stability)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.5
FONT_TITLE = 0.9

BAR_WIDTH = 200
BAR_GAP = 26

EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]

THRESHOLDS = {
    "angry": 35,
    "sad": 35,
    "happy": 35,
    "neutral": 35,
    "surprise": 70   # very high to prevent dominance
}
# =========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

# -------- FULLSCREEN WINDOW --------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------- SMOOTHING BUFFERS --------
emotion_history = {e: deque(maxlen=SMOOTHING_FRAMES) for e in EMOTIONS}

def smooth_scores(scores):
    for e in EMOTIONS:
        emotion_history[e].append(scores.get(e, 0))
    return {
        e: sum(emotion_history[e]) / len(emotion_history[e])
        for e in EMOTIONS
    }

frame_count = 0
last_result = None

stable_emotion = "neutral"
stable_conf = 0
last_switch_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    try:
        # -------- RUN DEEPFACE LESS FREQUENTLY --------
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0 or last_result is None:
            small = cv2.resize(frame, (320, 320))
            result = DeepFace.analyze(
                small,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )
            last_result = result[0] if isinstance(result, list) else result

        # -------- EMOTION PROCESSING --------
        emotion_scores = last_result["emotion"]
        smooth_emotions = smooth_scores(emotion_scores)

        current_top = max(smooth_emotions, key=smooth_emotions.get)
        current_conf = smooth_emotions[current_top]

        now = time.time()

        if (
            current_conf >= THRESHOLDS.get(current_top, 40)
            and current_top != stable_emotion
            and (now - last_switch_time) >= EMOTION_HOLD_SECONDS
        ):
            stable_emotion = current_top
            stable_conf = int(current_conf)
            last_switch_time = now

        top_emotion = stable_emotion
        top_conf = stable_conf

        # -------- FACE BOX (FOLLOW FACE) --------
        if "region" in last_result:
            r = last_result["region"]
            sx = w / 320
            sy = h / 320

            x = int(r["x"] * sx)
            y = int(r["y"] * sy)
            fw = int(r["w"] * sx)
            fh = int(r["h"] * sy)
        else:
            x, y, fw, fh = w // 4, h // 4, w // 2, h // 2

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 255, 255), 2)

        label = f"{top_emotion} ({top_conf}%)"
        cv2.rectangle(frame, (x, y - 28), (x + fw, y), (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 6, y - 8),
                    FONT, FONT_SMALL, (255, 255, 255), 1)

        # -------- HUD HEADER --------
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Facial Expression Intelligence System",
                    (20, 40), FONT, FONT_TITLE, (255, 0, 255), 2)

        # -------- EMOTION BARS --------
        start_x = 20
        start_y = 90

        for i, e in enumerate(EMOTIONS):
            y_pos = start_y + i * BAR_GAP
            value = int(smooth_emotions[e])

            cv2.putText(frame, e,
                        (start_x, y_pos),
                        FONT, FONT_SMALL, (255, 255, 255), 1)

            cv2.rectangle(frame,
                          (start_x + 80, y_pos - 10),
                          (start_x + 80 + BAR_WIDTH, y_pos),
                          (60, 60, 60), -1)

            fill = int((value / 100) * BAR_WIDTH)
            color = (255, 0, 255) if e == top_emotion else (180, 180, 180)

            cv2.rectangle(frame,
                          (start_x + 80, y_pos - 10),
                          (start_x + 80 + fill, y_pos),
                          color, -1)

        cv2.putText(frame, "Press 'q' to quit",
                    (w - 220, h - 20),
                    FONT, FONT_SMALL, (200, 200, 200), 1)

    except Exception:
        pass

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
