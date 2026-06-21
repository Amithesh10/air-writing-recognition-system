# ✍️ Air Writing Recognition

Draw letters and digits in the air with your fingertip — no stylus, no green
marker, no special hardware. A webcam and your index finger are all it takes.
[MediaPipe](https://github.com/google/mediapipe) tracks your hand in real
time, a CNN reads what you wrote, and the whole thing runs as a single
Streamlit app in the browser.

> Raise your index finger to draw. Lower it to lift the pen. Hit **Detect**
> to recognize the character.

<!-- Add a screenshot or GIF here once you have one, e.g.: -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Features

- **Fingertip tracking, no markers needed** — MediaPipe Hands detects 21
  hand landmarks per frame; the index fingertip is the pen.
- **Gesture-gated drawing** — raise your index finger to draw, lower it to
  pause. A peace-sign or open palm won't accidentally draw.
- **62-class character recognition** — a CNN trained on digits (0–9) and
  both cases of the Latin alphabet (A–Z, a–z).
- **Word building** — detected characters accumulate into a word, with
  Space / Backspace / Clear / Reset controls.
- **Per-character confidence** — every detected letter shows its model
  confidence, both live and in the running history.
- **Runs in the browser** — no desktop GUI windows; works locally or
  deployed to [Streamlit Community Cloud](https://streamlit.io/cloud).

## How it works

```
Webcam frame
     │
     ▼
MediaPipe Hands  ──▶  21 hand landmarks
     │
     ▼
Index finger up? ──▶  pen down → append fingertip point to stroke
     │
     ▼
Stroke trail drawn onto an in-memory "blackboard" image
     │
     ▼
Click Detect ──▶ crop → threshold → resize to 28×28 ──▶ CNN ──▶ character + confidence
     │
     ▼
Append to word
```

The CNN is a small architecture (2× Conv2D → MaxPool → Dropout → Dense →
Dropout → Dense-softmax) trained on 28×28 grayscale character images,
loaded from `model_saves/cnn_model.json` + `cnn_model_weights.h5`.

## Quick start (local)

```bash
git clone <your-repo-url>
cd air-writing-recognition-system
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), click
**START** under the camera panel, and allow camera access.

### Requirements

- Python 3.11 (recommended — matches the pinned dependency versions)
- A webcam
- See [`requirements.txt`](requirements.txt) for the full Python dependency
  list (Streamlit, streamlit-webrtc, MediaPipe, OpenCV, TensorFlow/Keras)

## Controls

| Action | Control |
|---|---|
| Draw | Raise index finger (keep other fingers down) |
| Pause / lift pen | Lower index finger |
| Recognize the current character | **🔍 Detect** button |
| Clear the blackboard without detecting | **🗑️ Clear board** |
| Add a space | **␣ Space** |
| Remove the last character | **⌫ Backspace** |
| Start over | **↺ Reset word** |

## Deploying to Streamlit Community Cloud

This repo deploys as-is. Two files matter beyond `streamlit_app.py`:

- **`requirements.txt`** — uses `opencv-python-headless` rather than the
  GUI build of OpenCV, since Cloud containers have no display.
- **`packages.txt`** — installs `libgl1`, a system library MediaPipe needs
  at runtime that isn't included in Streamlit Cloud's base image by
  default.

If you fork this and add new dependencies, keep both files in mind —
MediaPipe in particular pulls in a non-headless OpenCV build transitively,
which is the most common source of `ImportError: libGL.so.1` on this
platform. See **Troubleshooting** below if you hit deployment errors.

## Notes on the model file

`cnn_model.json` was saved with an old Keras (2.2.4 / TF1-era) format.
If you're running this with a newer Keras 3 environment, `model_from_json`
can't deserialize that schema directly and will raise
`Could not locate class 'Sequential'`. This app works around that by
rebuilding the same architecture in code and loading weights directly
from the `.h5` file by layer name — verified to reproduce the original
architecture exactly (same per-layer weight shapes, valid 62-class
softmax output). The pinned `requirements.txt` in this repo uses
Keras 2.15, where this isn't an issue, but the workaround is harmless
either way.

## Troubleshooting

**`ImportError: libGL.so.1: cannot open shared object file` (Streamlit
Community Cloud)**

This happens because `mediapipe` depends on `opencv-contrib-python` (the
full, non-headless OpenCV build) no matter what you put in
`requirements.txt` — and Streamlit Cloud's container is missing the system
graphics library that build needs. The fix is `packages.txt` (included in
this repo, at the same level as `streamlit_app.py`):

```
libgl1
```

Streamlit Cloud reads this file and runs `apt-get install` on each line
before installing Python packages.

**Don't add `libglib2.0-0` to this file.** It looks like the natural next
package to add (and many older tutorials list it), but as of mid-2026,
Streamlit Community Cloud's base image can't resolve it — `libglib2.0-0`
depends on `libffi7`/`libpcre3`, which no longer exist on the image, so
apt fails with "unmet dependencies" and your *entire* `packages.txt`
install aborts (including `libgl1`). This is a known platform limitation,
not something fixable from your repo. Keep `packages.txt` to just `libgl1`.

**If `libgl1` alone clears the `libGL.so.1` error but you then see
`ImportError: libgthread-2.0.so.0: cannot open shared object file`** — this
is the next layer of the same problem: `libgthread` normally ships inside
`libglib2.0-0`, which (per above) Streamlit Cloud can't install right now.
If you hit this, there isn't a `packages.txt`-only fix available on
Community Cloud at the moment. Options, roughly in order of effort:

1. Check the [Streamlit deployment issues board](https://discuss.streamlit.io/c/community-cloud/13)
   for an updated base image — this kind of platform issue tends to get
   fixed upstream, so it may already be resolved by the time you read this.
2. Self-host instead (any VPS/Docker host where you control the base image
   and can `apt-get install libglib2.0-0` freely — the local run
   instructions above work unmodified anywhere with a normal Linux/Mac/
   Windows Python install).
3. Deploy via Docker on a platform that lets you pick the base image
   (Render, Railway, Fly.io, Hugging Face Spaces, etc.) instead of
   Community Cloud's managed image.

**Camera doesn't start / black box**

Streamlit-webrtc needs HTTPS or `localhost` to get camera permission in
most browsers — `localhost` is fine for local use, and Streamlit Community
Cloud serves over HTTPS by default, so both work. A custom remote
deployment without TLS will not.

**Camera permission granted but video never appears (Streamlit Cloud
specifically)**

WebRTC sometimes needs a TURN relay server to traverse certain
networks/firewalls, and Streamlit Cloud doesn't provide one by default. If
START spins forever or the feed stays black only when deployed (but works
locally), see streamlit-webrtc's docs on configuring a free TURN server
(e.g. via Twilio's free tier) and passing it through `rtc_configuration`
in `webrtc_streamer(...)`.

**Laggy video**

Lower `min_detection_confidence` / `min_tracking_confidence` in
`AirWritingProcessor.__init__`, or reduce `FRAME_W` / `FRAME_H`.

**Detect button says "Nothing detected"**

Write a bigger / bolder character — the contour area threshold filters
out tiny smudges.

## Project structure

```
.
├── streamlit_app.py        # the app
├── requirements.txt        # Python dependencies (Cloud-safe, headless OpenCV)
├── packages.txt            # system (apt) dependencies for Streamlit Cloud
└── model_saves/
    ├── cnn_model.json      # model architecture
    └── cnn_model_weights.h5 # trained weights
```

## Why Streamlit instead of Flask

The original prototype used Flask with hand-rolled MJPEG video streaming
and a JS layer to wire buttons back to Python — every frame round-tripped
through an HTTP request/response cycle, which made the UI sluggish. This
version uses [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)
to open a real WebRTC connection from the browser's camera straight into a
background thread in the Python process. Video never touches Streamlit's
script reruns, so the drawing stays smooth; UI buttons just trigger a
normal Streamlit rerun that reads the latest state from that thread.

## Limitations

- Single-hand tracking only.
- Detection is a manual step (click **Detect**) rather than continuous
  recognition — this keeps the model from firing mid-stroke.
- Works best with large, clearly separated characters; very small or
  overlapping strokes may not produce a clean contour to crop.
- WebRTC camera access requires HTTPS or `localhost` in most browsers.

## License

# ✍️ Air Writing Recognition

Draw letters and digits in the air with your fingertip — no stylus, no green
marker, no special hardware. A webcam and your index finger are all it takes.
[MediaPipe](https://github.com/google/mediapipe) tracks your hand in real
time, a CNN reads what you wrote, and the whole thing runs as a single
Streamlit app in the browser.

> Raise your index finger to draw. Lower it to lift the pen. Hit **Detect**
> to recognize the character.

<!-- Add a screenshot or GIF here once you have one, e.g.: -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Features

- **Fingertip tracking, no markers needed** — MediaPipe Hands detects 21
  hand landmarks per frame; the index fingertip is the pen.
- **Gesture-gated drawing** — raise your index finger to draw, lower it to
  pause. A peace-sign or open palm won't accidentally draw.
- **62-class character recognition** — a CNN trained on digits (0–9) and
  both cases of the Latin alphabet (A–Z, a–z).
- **Word building** — detected characters accumulate into a word, with
  Space / Backspace / Clear / Reset controls.
- **Per-character confidence** — every detected letter shows its model
  confidence, both live and in the running history.
- **Runs in the browser** — no desktop GUI windows; works locally or
  deployed to [Streamlit Community Cloud](https://streamlit.io/cloud).

## How it works

```
Webcam frame
     │
     ▼
MediaPipe Hands  ──▶  21 hand landmarks
     │
     ▼
Index finger up? ──▶  pen down → append fingertip point to stroke
     │
     ▼
Stroke trail drawn onto an in-memory "blackboard" image
     │
     ▼
Click Detect ──▶ crop → threshold → resize to 28×28 ──▶ CNN ──▶ character + confidence
     │
     ▼
Append to word
```

The CNN is a small architecture (2× Conv2D → MaxPool → Dropout → Dense →
Dropout → Dense-softmax) trained on 28×28 grayscale character images,
loaded from `model_saves/cnn_model.json` + `cnn_model_weights.h5`.

## Quick start (local)

```bash
git clone <your-repo-url>
cd air-writing-recognition-system
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), click
**START** under the camera panel, and allow camera access.

### Requirements

- Python 3.11 (recommended — matches the pinned dependency versions)
- A webcam
- See [`requirements.txt`](requirements.txt) for the full Python dependency
  list (Streamlit, streamlit-webrtc, MediaPipe, OpenCV, TensorFlow/Keras)

## Controls

| Action | Control |
|---|---|
| Draw | Raise index finger (keep other fingers down) |
| Pause / lift pen | Lower index finger |
| Recognize the current character | **🔍 Detect** button |
| Clear the blackboard without detecting | **🗑️ Clear board** |
| Add a space | **␣ Space** |
| Remove the last character | **⌫ Backspace** |
| Start over | **↺ Reset word** |

## Deploying to Streamlit Community Cloud

This repo deploys as-is. Two files matter beyond `streamlit_app.py`:

- **`requirements.txt`** — uses `opencv-python-headless` rather than the
  GUI build of OpenCV, since Cloud containers have no display.
- **`packages.txt`** — installs `libgl1`, a system library MediaPipe needs
  at runtime that isn't included in Streamlit Cloud's base image by
  default.

If you fork this and add new dependencies, keep both files in mind —
MediaPipe in particular pulls in a non-headless OpenCV build transitively,
which is the most common source of `ImportError: libGL.so.1` on this
platform. See **Troubleshooting** below if you hit deployment errors.

## Notes on the model file

`cnn_model.json` was saved with an old Keras (2.2.4 / TF1-era) format.
If you're running this with a newer Keras 3 environment, `model_from_json`
can't deserialize that schema directly and will raise
`Could not locate class 'Sequential'`. This app works around that by
rebuilding the same architecture in code and loading weights directly
from the `.h5` file by layer name — verified to reproduce the original
architecture exactly (same per-layer weight shapes, valid 62-class
softmax output). The pinned `requirements.txt` in this repo uses
Keras 2.15, where this isn't an issue, but the workaround is harmless
either way.

## Troubleshooting

**`ImportError: libGL.so.1: cannot open shared object file` (Streamlit
Community Cloud)**

This happens because `mediapipe` depends on `opencv-contrib-python` (the
full, non-headless OpenCV build) no matter what you put in
`requirements.txt` — and Streamlit Cloud's container is missing the system
graphics library that build needs. The fix is `packages.txt` (included in
this repo, at the same level as `streamlit_app.py`):

```
libgl1
```

Streamlit Cloud reads this file and runs `apt-get install` on each line
before installing Python packages.

**Don't add `libglib2.0-0` to this file.** It looks like the natural next
package to add (and many older tutorials list it), but as of mid-2026,
Streamlit Community Cloud's base image can't resolve it — `libglib2.0-0`
depends on `libffi7`/`libpcre3`, which no longer exist on the image, so
apt fails with "unmet dependencies" and your *entire* `packages.txt`
install aborts (including `libgl1`). This is a known platform limitation,
not something fixable from your repo. Keep `packages.txt` to just `libgl1`.

**If `libgl1` alone clears the `libGL.so.1` error but you then see
`ImportError: libgthread-2.0.so.0: cannot open shared object file`** — this
is the next layer of the same problem: `libgthread` normally ships inside
`libglib2.0-0`, which (per above) Streamlit Cloud can't install right now.
If you hit this, there isn't a `packages.txt`-only fix available on
Community Cloud at the moment. Options, roughly in order of effort:

1. Check the [Streamlit deployment issues board](https://discuss.streamlit.io/c/community-cloud/13)
   for an updated base image — this kind of platform issue tends to get
   fixed upstream, so it may already be resolved by the time you read this.
2. Self-host instead (any VPS/Docker host where you control the base image
   and can `apt-get install libglib2.0-0` freely — the local run
   instructions above work unmodified anywhere with a normal Linux/Mac/
   Windows Python install).
3. Deploy via Docker on a platform that lets you pick the base image
   (Render, Railway, Fly.io, Hugging Face Spaces, etc.) instead of
   Community Cloud's managed image.

**Camera doesn't start / black box**

Streamlit-webrtc needs HTTPS or `localhost` to get camera permission in
most browsers — `localhost` is fine for local use, and Streamlit Community
Cloud serves over HTTPS by default, so both work. A custom remote
deployment without TLS will not.

**Camera permission granted but video never appears (Streamlit Cloud
specifically)**

WebRTC sometimes needs a TURN relay server to traverse certain
networks/firewalls, and Streamlit Cloud doesn't provide one by default. If
START spins forever or the feed stays black only when deployed (but works
locally), see streamlit-webrtc's docs on configuring a free TURN server
(e.g. via Twilio's free tier) and passing it through `rtc_configuration`
in `webrtc_streamer(...)`.

**Laggy video**

Lower `min_detection_confidence` / `min_tracking_confidence` in
`AirWritingProcessor.__init__`, or reduce `FRAME_W` / `FRAME_H`.

**Detect button says "Nothing detected"**

Write a bigger / bolder character — the contour area threshold filters
out tiny smudges.

## Project structure

```
.
├── streamlit_app.py        # the app
├── requirements.txt        # Python dependencies (Cloud-safe, headless OpenCV)
├── packages.txt            # system (apt) dependencies for Streamlit Cloud
└── model_saves/
    ├── cnn_model.json      # model architecture
    └── cnn_model_weights.h5 # trained weights
```

## Why Streamlit instead of Flask

The original prototype used Flask with hand-rolled MJPEG video streaming
and a JS layer to wire buttons back to Python — every frame round-tripped
through an HTTP request/response cycle, which made the UI sluggish. This
version uses [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)
to open a real WebRTC connection from the browser's camera straight into a
background thread in the Python process. Video never touches Streamlit's
script reruns, so the drawing stays smooth; UI buttons just trigger a
normal Streamlit rerun that reads the latest state from that thread.

## Limitations

- Single-hand tracking only.
- Detection is a manual step (click **Detect**) rather than continuous
  recognition — this keeps the model from firing mid-stroke.
- Works best with large, clearly separated characters; very small or
  overlapping strokes may not produce a clean contour to crop.
- WebRTC camera access requires HTTPS or `localhost` in most browsers.

## License

<!-- Add your license here, e.g. MIT -->

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for hand-landmark tracking
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for
  in-browser real-time video# ✍️ Air Writing Recognition

Draw letters and digits in the air with your fingertip — no stylus, no green
marker, no special hardware. A webcam and your index finger are all it takes.
[MediaPipe](https://github.com/google/mediapipe) tracks your hand in real
time, a CNN reads what you wrote, and the whole thing runs as a single
Streamlit app in the browser.

> Raise your index finger to draw. Lower it to lift the pen. Hit **Detect**
> to recognize the character.

<!-- Add a screenshot or GIF here once you have one, e.g.: -->
<!-- ![App screenshot](docs/screenshot.png) -->

## Features

- **Fingertip tracking, no markers needed** — MediaPipe Hands detects 21
  hand landmarks per frame; the index fingertip is the pen.
- **Gesture-gated drawing** — raise your index finger to draw, lower it to
  pause. A peace-sign or open palm won't accidentally draw.
- **62-class character recognition** — a CNN trained on digits (0–9) and
  both cases of the Latin alphabet (A–Z, a–z).
- **Word building** — detected characters accumulate into a word, with
  Space / Backspace / Clear / Reset controls.
- **Per-character confidence** — every detected letter shows its model
  confidence, both live and in the running history.
- **Runs in the browser** — no desktop GUI windows; works locally or
  deployed to [Streamlit Community Cloud](https://streamlit.io/cloud).

## How it works

```
Webcam frame
     │
     ▼
MediaPipe Hands  ──▶  21 hand landmarks
     │
     ▼
Index finger up? ──▶  pen down → append fingertip point to stroke
     │
     ▼
Stroke trail drawn onto an in-memory "blackboard" image
     │
     ▼
Click Detect ──▶ crop → threshold → resize to 28×28 ──▶ CNN ──▶ character + confidence
     │
     ▼
Append to word
```

The CNN is a small architecture (2× Conv2D → MaxPool → Dropout → Dense →
Dropout → Dense-softmax) trained on 28×28 grayscale character images,
loaded from `model_saves/cnn_model.json` + `cnn_model_weights.h5`.

## Quick start (local)

```bash
git clone <your-repo-url>
cd air-writing-recognition-system
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), click
**START** under the camera panel, and allow camera access.

### Requirements

- Python 3.11 (recommended — matches the pinned dependency versions)
- A webcam
- See [`requirements.txt`](requirements.txt) for the full Python dependency
  list (Streamlit, streamlit-webrtc, MediaPipe, OpenCV, TensorFlow/Keras)

## Controls

| Action | Control |
|---|---|
| Draw | Raise index finger (keep other fingers down) |
| Pause / lift pen | Lower index finger |
| Recognize the current character | **🔍 Detect** button |
| Clear the blackboard without detecting | **🗑️ Clear board** |
| Add a space | **␣ Space** |
| Remove the last character | **⌫ Backspace** |
| Start over | **↺ Reset word** |

## Deploying to Streamlit Community Cloud

This repo deploys as-is. Two files matter beyond `streamlit_app.py`:

- **`requirements.txt`** — uses `opencv-python-headless` rather than the
  GUI build of OpenCV, since Cloud containers have no display.
- **`packages.txt`** — installs `libgl1`, a system library MediaPipe needs
  at runtime that isn't included in Streamlit Cloud's base image by
  default.

If you fork this and add new dependencies, keep both files in mind —
MediaPipe in particular pulls in a non-headless OpenCV build transitively,
which is the most common source of `ImportError: libGL.so.1` on this
platform. See **Troubleshooting** below if you hit deployment errors.

## Notes on the model file

`cnn_model.json` was saved with an old Keras (2.2.4 / TF1-era) format.
If you're running this with a newer Keras 3 environment, `model_from_json`
can't deserialize that schema directly and will raise
`Could not locate class 'Sequential'`. This app works around that by
rebuilding the same architecture in code and loading weights directly
from the `.h5` file by layer name — verified to reproduce the original
architecture exactly (same per-layer weight shapes, valid 62-class
softmax output). The pinned `requirements.txt` in this repo uses
Keras 2.15, where this isn't an issue, but the workaround is harmless
either way.

## Troubleshooting

**`ImportError: libGL.so.1: cannot open shared object file` (Streamlit
Community Cloud)**

This happens because `mediapipe` depends on `opencv-contrib-python` (the
full, non-headless OpenCV build) no matter what you put in
`requirements.txt` — and Streamlit Cloud's container is missing the system
graphics library that build needs. The fix is `packages.txt` (included in
this repo, at the same level as `streamlit_app.py`):

```
libgl1
```

Streamlit Cloud reads this file and runs `apt-get install` on each line
before installing Python packages.

**Don't add `libglib2.0-0` to this file.** It looks like the natural next
package to add (and many older tutorials list it), but as of mid-2026,
Streamlit Community Cloud's base image can't resolve it — `libglib2.0-0`
depends on `libffi7`/`libpcre3`, which no longer exist on the image, so
apt fails with "unmet dependencies" and your *entire* `packages.txt`
install aborts (including `libgl1`). This is a known platform limitation,
not something fixable from your repo. Keep `packages.txt` to just `libgl1`.

**If `libgl1` alone clears the `libGL.so.1` error but you then see
`ImportError: libgthread-2.0.so.0: cannot open shared object file`** — this
is the next layer of the same problem: `libgthread` normally ships inside
`libglib2.0-0`, which (per above) Streamlit Cloud can't install right now.
If you hit this, there isn't a `packages.txt`-only fix available on
Community Cloud at the moment. Options, roughly in order of effort:

1. Check the [Streamlit deployment issues board](https://discuss.streamlit.io/c/community-cloud/13)
   for an updated base image — this kind of platform issue tends to get
   fixed upstream, so it may already be resolved by the time you read this.
2. Self-host instead (any VPS/Docker host where you control the base image
   and can `apt-get install libglib2.0-0` freely — the local run
   instructions above work unmodified anywhere with a normal Linux/Mac/
   Windows Python install).
3. Deploy via Docker on a platform that lets you pick the base image
   (Render, Railway, Fly.io, Hugging Face Spaces, etc.) instead of
   Community Cloud's managed image.

**Camera doesn't start / black box**

Streamlit-webrtc needs HTTPS or `localhost` to get camera permission in
most browsers — `localhost` is fine for local use, and Streamlit Community
Cloud serves over HTTPS by default, so both work. A custom remote
deployment without TLS will not.

**Camera permission granted but video never appears (Streamlit Cloud
specifically)**

WebRTC sometimes needs a TURN relay server to traverse certain
networks/firewalls, and Streamlit Cloud doesn't provide one by default. If
START spins forever or the feed stays black only when deployed (but works
locally), see streamlit-webrtc's docs on configuring a free TURN server
(e.g. via Twilio's free tier) and passing it through `rtc_configuration`
in `webrtc_streamer(...)`.

**Laggy video**

Lower `min_detection_confidence` / `min_tracking_confidence` in
`AirWritingProcessor.__init__`, or reduce `FRAME_W` / `FRAME_H`.

**Detect button says "Nothing detected"**

Write a bigger / bolder character — the contour area threshold filters
out tiny smudges.

## Project structure

```
.
├── streamlit_app.py        # the app
├── requirements.txt        # Python dependencies (Cloud-safe, headless OpenCV)
├── packages.txt            # system (apt) dependencies for Streamlit Cloud
└── model_saves/
    ├── cnn_model.json      # model architecture
    └── cnn_model_weights.h5 # trained weights
```

## Why Streamlit instead of Flask

The original prototype used Flask with hand-rolled MJPEG video streaming
and a JS layer to wire buttons back to Python — every frame round-tripped
through an HTTP request/response cycle, which made the UI sluggish. This
version uses [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)
to open a real WebRTC connection from the browser's camera straight into a
background thread in the Python process. Video never touches Streamlit's
script reruns, so the drawing stays smooth; UI buttons just trigger a
normal Streamlit rerun that reads the latest state from that thread.

## Limitations

- Single-hand tracking only.
- Detection is a manual step (click **Detect**) rather than continuous
  recognition — this keeps the model from firing mid-stroke.
- Works best with large, clearly separated characters; very small or
  overlapping strokes may not produce a clean contour to crop.
- WebRTC camera access requires HTTPS or `localhost` in most browsers.

## License

Apache 2.0

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for hand-landmark tracking
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for
  in-browser real-time video

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for hand-landmark tracking
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for
  in-browser real-time video
