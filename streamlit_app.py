import time

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Blink a Word", layout="wide")
st.title("Blink to Select Word Game")
st.caption(
    "Allow camera access, place the pink dot (between your eyes) over a scrolling "
    "word, and blink to select it. Select all words in order for a bonus!"
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SENTENCE = "Welcome to the blink to select word game"
WORDS = SENTENCE.split()
BLINK_THRESHOLD = 1.5
BLINK_COOLDOWN = 1.5
POINTS_PER_WORD = 10
BONUS_POINTS = 50
GAME_DURATION = 30

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(pts):
    top, bottom = pts[1], pts[5]
    left, right = pts[0], pts[3]
    vertical = ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5
    horizontal = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5
    return horizontal / vertical if vertical != 0 else 0


class GameProcessor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.reset_requested = True
        self.frame_w = None

    def _reset(self, w, h):
        self.frame_w = w
        self.font_scale = 1.2 * w / 1280
        self.word_gap = max(int(40 * w / 1280), 15)
        self.y_position = h // 2
        self.selected_words = []
        self.points = 0
        self.last_blink_time = 0.0
        self.start_time = time.time()
        self.word_positions = []
        current_x = w
        for word in WORDS:
            word_width = cv2.getTextSize(word, FONT, self.font_scale, 2)[0][0]
            self.word_positions.append([word, current_x, word_width])
            current_x += word_width + self.word_gap
        self.reset_requested = False

    def _scaled_text(self, img, text, pos, scale, color, thickness):
        cv2.putText(img, text, pos, FONT, scale * self.frame_w / 1280, color, thickness)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        if self.reset_requested or self.frame_w != w:
            self._reset(w, h)

        elapsed_time = time.time() - self.start_time
        remaining_time = int(GAME_DURATION - elapsed_time)

        if remaining_time <= 0:
            self._scaled_text(img, "Time's up!", (int(w * 0.39), self.y_position), 2, (0, 0, 255), 4)
            self._scaled_text(img, f"Final Score: {self.points}", (int(w * 0.37), self.y_position + int(h * 0.08)), 1.5, (0, 255, 255), 3)
            self._scaled_text(img, "Press 'Restart game' below to play again", (int(w * 0.25), self.y_position + int(h * 0.17)), 0.8, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            pt = lambda i: (int(lms[i].x * w), int(lms[i].y * h))
            left_eye = [pt(i) for i in LEFT_EYE]
            right_eye = [pt(i) for i in RIGHT_EYE]
            avg_ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            cx, cy = pt(168)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)

            if avg_ratio > BLINK_THRESHOLD and time.time() - self.last_blink_time > BLINK_COOLDOWN:
                self.last_blink_time = time.time()
                for word, x, word_width in self.word_positions:
                    if x - 10 < cx < x + word_width + 10 and self.y_position - 40 < cy < self.y_position + 40:
                        if word not in self.selected_words:
                            self.selected_words.append(word)
                            self.points += POINTS_PER_WORD
                            if self.selected_words == WORDS:
                                self.points += BONUS_POINTS
                            break

        speed = 2 + int(elapsed_time / 5)
        for i in range(len(self.word_positions)):
            word, x, word_width = self.word_positions[i]
            x -= speed
            if x + word_width < 0:
                max_x = max(p[1] + p[2] for p in self.word_positions)
                x = max(max_x + self.word_gap, w)
            self.word_positions[i][1] = x

        for word, x, _ in self.word_positions:
            color = (0, 255, 0) if word in self.selected_words else (255, 255, 255)
            cv2.putText(img, word, (int(x), self.y_position), FONT, self.font_scale, color, 2)

        self._scaled_text(img, f"Points: {self.points}", (20, int(h * 0.07)), 1, (0, 200, 255), 2)
        self._scaled_text(img, f"Time left: {remaining_time}s", (int(w * 0.75), int(h * 0.07)), 1, (255, 100, 100), 2)
        self._scaled_text(img, "Selected: " + " ".join(self.selected_words), (20, int(h * 0.95)), 0.8, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def get_ice_servers():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        username = st.secrets["TURN_USERNAME"]
        credential = st.secrets["TURN_CREDENTIAL"]
        servers.append(
            {
                "urls": [
                    "turn:standard.relay.metered.ca:80",
                    "turn:standard.relay.metered.ca:443",
                    "turns:standard.relay.metered.ca:443?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            }
        )
    except (KeyError, FileNotFoundError):
        st.warning("No TURN credentials set - video may fail on restrictive networks.")
    return servers


ctx = webrtc_streamer(
    key="blink-a-word",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GameProcessor,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor and st.button("Restart game"):
    ctx.video_processor.reset_requested = True
