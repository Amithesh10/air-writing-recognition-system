# ============================================================
# AIR WRITING CHARACTER + WORD RECOGNITION — Streamlit version
# ============================================================
#
# WHY STREAMLIT INSTEAD OF FLASK:
#   Flask needs you to hand-roll video streaming (MJPEG generators,
#   manual <img> tag refresh) and a separate JS layer to talk back
#   to Python for button clicks. It's slow because every frame is
#   round-tripped through Flask's request/response cycle.
#
#   Streamlit + streamlit-webrtc instead opens a real WebRTC peer
#   connection straight from the browser's camera to a background
#   thread in this process. Frames never touch Streamlit's script
#   reruns, so drawing stays smooth. Buttons just trigger a normal
#   Streamlit rerun, which reads the latest state out of that
#   background thread.
#
# INSTALL:
#   pip install streamlit streamlit-webrtc mediapipe opencv-python-headless tensorflow av
#
# RUN:
#   streamlit run streamlit_app.py
#
# FILES NEEDED (same as before):
#   model_saves/cnn_model.json
#   model_saves/cnn_model_weights.h5
# ============================================================

import threading
import os
import time

import av
import cv2
import h5py
import numpy as np
import streamlit as st
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Air Writing Recognition",
    page_icon="✍️",
    layout="wide",
)

# ============================================================
# THEME
# ============================================================
# A chalkboard-and-ink system: the app's whole job is turning a
# fingertip trail into characters, so the palette stays close to
# that material world — slate board, chalk cream, and two accent
# colors that map directly onto app state (live/pen-down vs a
# confirmed detection) rather than decoration.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --board:        #1B1F23;
        --board-raised: #23282F;
        --panel:        #2A2F36;
        --chalk:        #F5F1E8;
        --chalk-dim:    #C9C2B4;
        --slate:        #8B95A1;
        --live:         #FF6B4A;
        --confirmed:    #5FD4A0;
        --hairline:     rgba(245, 241, 232, 0.10);
    }

    .stApp {
        background: var(--board);
    }

    /* Faint chalk-dust texture on the page background */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.035;
        background-image:
            radial-gradient(circle at 20% 30%, var(--chalk) 0.5px, transparent 0.5px),
            radial-gradient(circle at 70% 65%, var(--chalk) 0.5px, transparent 0.5px),
            radial-gradient(circle at 45% 80%, var(--chalk) 0.5px, transparent 0.5px);
        background-size: 140px 140px, 180px 180px, 120px 120px;
        z-index: 0;
    }

    section.main > div { padding-top: 1.5rem; }

    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--chalk) !important;
        letter-spacing: -0.01em;
    }

    p, span, label, div { color: var(--chalk-dim); }

    /* ---------- Hero ---------- */
    .hero {
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin-bottom: 2px;
    }
    .hero-mark {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        color: var(--live);
        background: rgba(255, 107, 74, 0.12);
        border: 1px solid rgba(255, 107, 74, 0.35);
        padding: 3px 9px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--chalk);
        margin: 0;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--slate);
        margin-top: 6px;
        margin-bottom: 1.6rem;
        max-width: 640px;
        line-height: 1.5;
    }
    .hero-sub b { color: var(--chalk-dim); font-weight: 600; }

    /* ---------- Section labels (eyebrow style) ---------- */
    .rail-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--slate);
        margin: 1.4rem 0 0.55rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .rail-label:first-child { margin-top: 0; }
    .rail-label::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--hairline);
    }

    /* ---------- Word display (the signature element) ---------- */
    .word-stage {
        background: var(--board-raised);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 18px 20px;
        min-height: 86px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .word-empty {
        font-family: 'JetBrains Mono', monospace;
        color: var(--slate);
        font-size: 0.88rem;
        font-style: normal;
    }
    .char-tile {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--chalk);
        background: rgba(245, 241, 232, 0.05);
        border-radius: 6px;
        padding: 4px 9px 8px 9px;
        position: relative;
        min-width: 1.1em;
        text-align: center;
    }
    .char-tile.space-tile {
        color: var(--slate);
        font-size: 1rem;
        padding-bottom: 12px;
    }
    .char-tile .conf-bar {
        position: absolute;
        left: 6px;
        right: 6px;
        bottom: 4px;
        height: 3px;
        border-radius: 2px;
        background: var(--hairline);
        overflow: hidden;
    }
    .char-tile .conf-bar-fill {
        height: 100%;
        background: var(--confirmed);
        border-radius: 2px;
    }
    .cursor-blink {
        display: inline-block;
        width: 2px;
        height: 1.55rem;
        background: var(--live);
        animation: blink 1.1s step-end infinite;
        align-self: center;
        margin-left: 2px;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ---------- Status pill ---------- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 6px 12px;
        border-radius: 20px;
        margin-bottom: 0.9rem;
    }
    .status-pill.live {
        background: rgba(255, 107, 74, 0.13);
        color: var(--live);
        border: 1px solid rgba(255, 107, 74, 0.4);
    }
    .status-pill.idle {
        background: rgba(139, 149, 161, 0.12);
        color: var(--slate);
        border: 1px solid rgba(139, 149, 161, 0.3);
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: currentColor;
    }
    .status-pill.live .status-dot { animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(255,107,74,0.5); }
        50% { opacity: 0.6; box-shadow: 0 0 0 5px rgba(255,107,74,0); }
    }

    /* ---------- Prediction readout ---------- */
    .predict-card {
        background: var(--board-raised);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .predict-glyph {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--confirmed);
        line-height: 1;
        min-width: 44px;
        text-align: center;
    }
    .predict-meta { flex: 1; }
    .predict-conf-label {
        font-size: 0.75rem;
        color: var(--slate);
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }
    .predict-conf-track {
        height: 5px;
        background: var(--hairline);
        border-radius: 3px;
        overflow: hidden;
    }
    .predict-conf-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--live), var(--confirmed));
        border-radius: 3px;
    }
    .predict-empty {
        color: var(--slate);
        font-size: 0.88rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------- History strip ---------- */
    .history-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .history-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--chalk-dim);
        background: rgba(245, 241, 232, 0.05);
        border: 1px solid var(--hairline);
        border-radius: 5px;
        padding: 3px 8px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
        border-radius: 8px !important;
        border: 1px solid var(--hairline) !important;
        background: var(--board-raised) !important;
        color: var(--chalk) !important;
        transition: border-color 0.15s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        border-color: var(--live) !important;
        color: var(--live) !important;
    }
    .stButton > button:active { transform: scale(0.97); }
    .stButton > button[kind="primary"] {
        background: var(--live) !important;
        border-color: var(--live) !important;
        color: var(--board) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #ff7d61 !important;
        color: var(--board) !important;
    }

    /* ---------- Misc Streamlit cleanup ---------- */
    [data-testid="stExpander"] {
        background: var(--board-raised);
        border: 1px solid var(--hairline);
        border-radius: 10px;
    }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--slate) !important; }
    hr { border-color: var(--hairline) !important; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CHARACTER LABELS  (0-9, A-Z, a-z  →  62 classes)
# ============================================================

CHARACTERS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',
    41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o',
    51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
    61: 'z',
}

MODEL_JSON_PATH = os.path.join("model_saves", "cnn_model.json")
MODEL_WEIGHTS_PATH = os.path.join("model_saves", "cnn_model_weights.h5")

FRAME_W, FRAME_H = 640, 480

IDX_TIP, IDX_PIP = 8, 6     # index fingertip / PIP joint
MID_TIP, MID_PIP = 12, 10   # middle fingertip / PIP joint (peace-sign guard)


# ============================================================
# MODEL LOADING
# ============================================================
#
# The shipped cnn_model.json is an old Keras 2.2.4 / TF1-style
# config. Modern Keras 3's model_from_json() can't deserialize
# that schema directly ("Could not locate class 'Sequential'").
# Rather than fight version skew, we rebuild the same architecture
# by hand (it's a small, fixed 8-layer CNN) and load weights
# straight out of the .h5 file by layer name by name using h5py,
# bypassing Keras's own H5 loader. Verified to reproduce the
# original architecture's output shape (1, 62) and a valid
# softmax distribution.
# ============================================================

@st.cache_resource(show_spinner="Loading CNN model...")
def load_cnn_model():
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_1'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        layers.Dropout(0.25, name='dropout_1'),
        layers.Flatten(name='flatten_1'),
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.Dropout(0.5, name='dropout_2'),
        layers.Dense(62, activation='softmax', name='dense_2'),
    ])

    with h5py.File(MODEL_WEIGHTS_PATH, "r") as f:
        for lname in ['conv2d_1', 'conv2d_2', 'dense_1', 'dense_2']:
            grp = f['model_weights'][lname][lname]
            kernel = np.array(grp['kernel:0'])
            bias = np.array(grp['bias:0'])
            model.get_layer(lname).set_weights([kernel, bias])

    # Warm up the graph once at load time (not on the user's first click).
    # model.predict() has ~50-60ms of fixed per-call overhead (it spins up
    # a tf.data pipeline internally every time); calling the model directly
    # skips that and is ~6-8x faster. We use __call__ everywhere below.
    _ = model(np.zeros((1, 28, 28, 1), dtype="float32"), training=False)

    return model


def predict_character(model, gray_28x28):
    img = gray_28x28.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    probs = model(img, training=False).numpy()[0]
    idx = int(np.argmax(probs))
    return CHARACTERS[idx], float(probs[idx])


def safe_crop(image, x, y, w, h, pad=12):
    H, W = image.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    return image[y1:y2, x1:x2]


def extract_and_predict(model, blackboard):
    gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    blur = cv2.medianBlur(blur, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None, 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    char_img = safe_crop(gray, x, y, w, h)
    if char_img.size == 0:
        return None, 0.0

    resized = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
    return predict_character(model, resized)


def is_index_up(hand_landmarks):
    tip = hand_landmarks.landmark[IDX_TIP]
    pip = hand_landmarks.landmark[IDX_PIP]
    mid_tip = hand_landmarks.landmark[MID_TIP]
    mid_pip = hand_landmarks.landmark[MID_PIP]
    index_extended = tip.y < pip.y
    middle_down = mid_tip.y > mid_pip.y
    return index_extended and middle_down


# ============================================================
# VIDEO PROCESSOR
# ============================================================
# Runs in a background thread managed by streamlit-webrtc, once
# per incoming camera frame. All state that the main Streamlit
# script needs to read (blackboard, pen status) is guarded by a
# lock since the two threads run concurrently.
# ============================================================

class AirWritingProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.pts = deque(maxlen=512)
        self.blackboard = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.pen_down = False

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.clear_requested = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (FRAME_W, FRAME_H))

        with self.lock:
            if self.clear_requested:
                self.blackboard = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                self.pts = deque(maxlen=512)
                self.clear_requested = False

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        rgb.flags.writeable = True

        pen_down = False
        fingertip = None

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]

            self.mp_drawing.draw_landmarks(
                img,
                hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            lm8 = hand_lm.landmark[IDX_TIP]
            fx, fy = int(lm8.x * img.shape[1]), int(lm8.y * img.shape[0])
            fingertip = (fx, fy)

            pen_down = is_index_up(hand_lm)

            # BGR order. Live coral #FF6B4A -> (74,107,255); idle slate #8B95A1 -> (161,149,139)
            tip_color = (74, 107, 255) if pen_down else (161, 149, 139)
            cv2.circle(img, fingertip, 10, tip_color, -1)
            cv2.circle(img, fingertip, 14, tip_color, 2)

            with self.lock:
                if pen_down:
                    self.pts.appendleft(fingertip)
                else:
                    self.pts.appendleft(None)
        else:
            with self.lock:
                self.pts.appendleft(None)

        with self.lock:
            self.pen_down = pen_down
            pts_snapshot = list(self.pts)
            for i in range(1, len(pts_snapshot)):
                if pts_snapshot[i - 1] is None or pts_snapshot[i] is None:
                    continue
                cv2.line(img, pts_snapshot[i - 1], pts_snapshot[i], (74, 107, 255), 6)
                cv2.line(self.blackboard, pts_snapshot[i - 1], pts_snapshot[i], (255, 255, 255), 8)

        pen_label = "PEN DOWN" if pen_down else "PEN UP"
        # BGR order. Live accent #FF6B4A -> (74,107,255); chalk-dim #C9C2B4 -> (180,194,201)
        pen_color = (74, 107, 255) if pen_down else (180, 194, 201)
        cv2.rectangle(img, (0, 0), (190, 34), (31, 27, 23), -1)
        cv2.putText(img, pen_label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, pen_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_blackboard_copy(self):
        with self.lock:
            return self.blackboard.copy()

    def request_clear(self):
        with self.lock:
            self.clear_requested = True

    def is_pen_down(self):
        with self.lock:
            return self.pen_down


# ============================================================
# SESSION STATE
# ============================================================

if "word" not in st.session_state:
    st.session_state.word = ""
if "pred_char" not in st.session_state:
    st.session_state.pred_char = ""
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "history" not in st.session_state:
    st.session_state.history = []  # list of (char, confidence)

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="hero-mark">CNN · 62 CLASSES</span>
    </div>
    <h1 class="hero-title">Air Writing Recognition</h1>
    <p class="hero-sub">
        Raise your <b>index finger</b> to draw in the air — lower it to lift the pen.
        Trace a character on the board, then hit <b>Detect</b> to read it.
    </p>
    """,
    unsafe_allow_html=True,
)

model = load_cnn_model()

col_video, col_panel = st.columns([3, 2], gap="large")

with col_video:
    st.markdown('<div class="rail-label">Camera</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="air-writing",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AirWritingProcessor,
        media_stream_constraints={
            "video": {"width": FRAME_W, "height": FRAME_H},
            "audio": False,
        },
        async_processing=True,
    )

    status_slot = st.empty()

    st.markdown('<div class="rail-label">Blackboard — fed to the model</div>', unsafe_allow_html=True)
    blackboard_slot = st.empty()

with col_panel:
    st.markdown('<div class="rail-label">Recognized word</div>', unsafe_allow_html=True)
    word_display = st.empty()

    st.markdown('<div class="rail-label">Controls</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    b3, b4 = st.columns(2)

    detect_clicked = b1.button("🔍 Detect", use_container_width=True, type="primary")
    clear_clicked = b2.button("🗑️ Clear board", use_container_width=True)
    space_clicked = b3.button("␣ Space", use_container_width=True)
    backspace_clicked = b4.button("⌫ Backspace", use_container_width=True)

    reset_word_clicked = st.button("↺ Reset word", use_container_width=True)

    st.markdown('<div class="rail-label">Last detection</div>', unsafe_allow_html=True)
    pred_display = st.empty()

    st.markdown('<div class="rail-label">History</div>', unsafe_allow_html=True)
    history_display = st.empty()

# ============================================================
# BUTTON LOGIC
# ============================================================

if ctx.video_processor:
    if clear_clicked:
        ctx.video_processor.request_clear()
        st.session_state.pred_char = ""

    if space_clicked:
        st.session_state.word += " "

    if backspace_clicked:
        st.session_state.word = st.session_state.word[:-1]

    if reset_word_clicked:
        st.session_state.word = ""
        st.session_state.pred_char = ""
        st.session_state.confidence = 0.0
        st.session_state.history = []

    if detect_clicked:
        bb = ctx.video_processor.get_blackboard_copy()
        char, conf = extract_and_predict(model, bb)
        if char:
            st.session_state.pred_char = char
            st.session_state.confidence = conf
            st.session_state.word += char
            st.session_state.history.append((char, conf))
            st.toast(f"Detected '{char}' ({conf*100:.0f}%)", icon="✅")
        else:
            st.toast("Nothing detected — write a bigger character and try again.", icon="⚠️")
        ctx.video_processor.request_clear()

# ============================================================
# RENDER PANELS
# ============================================================

# --- Pen status pill (reads current pen state from the processor) ---
if ctx.video_processor:
    pen_is_down = ctx.video_processor.is_pen_down()
    if pen_is_down:
        status_slot.markdown(
            '<div class="status-pill live"><span class="status-dot"></span>'
            'PEN DOWN — drawing</div>',
            unsafe_allow_html=True,
        )
    else:
        status_slot.markdown(
            '<div class="status-pill idle"><span class="status-dot"></span>'
            'PEN UP — raise index finger to draw</div>',
            unsafe_allow_html=True,
        )
else:
    status_slot.markdown(
        '<div class="status-pill idle"><span class="status-dot"></span>'
        'Camera not started</div>',
        unsafe_allow_html=True,
    )

# --- Word stage: each detected character is a chalk tile with a
#     confidence underline; spaces render as a thin gap glyph ---
if st.session_state.word:
    tiles = []
    for ch, conf in st.session_state.history:
        pct = max(0.0, min(1.0, conf)) * 100
        tiles.append(
            f'<div class="char-tile">{ch}'
            f'<div class="conf-bar"><div class="conf-bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'</div>'
        )
    # account for plain spaces added via the Space button (not in history)
    extra_spaces = len(st.session_state.word) - len(st.session_state.history)
    if extra_spaces > 0:
        tiles.append('<div class="char-tile space-tile">␣</div>' * extra_spaces)

    word_display.markdown(
        f'<div class="word-stage">{"".join(tiles)}<span class="cursor-blink"></span></div>',
        unsafe_allow_html=True,
    )
else:
    word_display.markdown(
        '<div class="word-stage"><span class="word-empty">Nothing written yet — draw a character and hit Detect.</span></div>',
        unsafe_allow_html=True,
    )

# --- Last prediction card ---
if st.session_state.pred_char:
    pct = st.session_state.confidence * 100
    pred_display.markdown(
        f"""
        <div class="predict-card">
            <div class="predict-glyph">{st.session_state.pred_char}</div>
            <div class="predict-meta">
                <div class="predict-conf-label">{pct:.1f}% CONFIDENCE</div>
                <div class="predict-conf-track">
                    <div class="predict-conf-fill" style="width:{pct:.0f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    pred_display.markdown(
        '<div class="predict-card"><span class="predict-empty">No detection yet.</span></div>',
        unsafe_allow_html=True,
    )

# --- History chips ---
if st.session_state.history:
    chips = "".join(
        f'<span class="history-chip">{c} · {conf*100:.0f}%</span>'
        for c, conf in st.session_state.history[-15:]
    )
    history_display.markdown(f'<div class="history-strip">{chips}</div>', unsafe_allow_html=True)
else:
    history_display.markdown('<span class="word-empty">Empty.</span>', unsafe_allow_html=True)

# --- Blackboard preview ---
if ctx.video_processor:
    bb = ctx.video_processor.get_blackboard_copy()
    blackboard_slot.image(cv2.cvtColor(bb, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    blackboard_slot.info("Click **START** above to enable the camera.")

with st.expander("ℹ️ How to use this app"):
    st.markdown(
        """
1. Click **START** under the camera feed and allow browser camera access.
2. Raise your **index finger** (other fingers down) to draw — keep middle finger lowered or it's treated as a non-drawing gesture (e.g. peace sign).
3. Lower your finger to lift the pen / pause the stroke.
4. When you've written one character on the blackboard, click **🔍 Detect**.
   This crops the drawing, resizes it to 28×28, and runs it through the CNN.
5. Click **␣ Space** between words, **⌫ Backspace** to remove the last character,
   and **🗑️ Clear board** if a stroke goes wrong before detecting.
6. **↺ Reset word** clears everything and starts over.

**Why a button instead of a keyboard shortcut?** Browsers don't reliably hand raw
keypresses to background camera threads the way a native OpenCV window did, so
detection is button-driven here instead of the original script's `D` key.
        """
    )