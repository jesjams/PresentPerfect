# ──────────────────────────────  Imports & Monkey-patch  ──────────────────────────────
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, send_from_directory, jsonify
from pathlib import Path
from flask_socketio import SocketIO
from flask_cors import CORS

import os, time, tempfile, random, queue, threading, math, subprocess
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, wait

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO
import whisper

from dotenv import load_dotenv
from typing import Dict, List, Any
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI

from itertools import chain         
import librosa
import numpy as np       
from text_metrics import get_metrics
from whisper_helper import whisper_transcriber
from text_to_speech import get_speech
from enhanced_transcript import enhance_transcript_for_presentation
from vocal_dynamics_analyzer import analyze_vocal_dynamics

video_file_tracker = {}
user = ""

torch.backends.cudnn.benchmark = True
# DEVICE = "cuda:0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_dotenv()  
#Flask / Socket.IO  
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=120, ping_interval=25 )

STATIC_AUDIO_DIR = os.path.join(os.getcwd(), 'static', 'generated_audio')
os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)
print(f"[INFO] Static audio directory: {STATIC_AUDIO_DIR}")
print(f"[INFO] Directory exists: {os.path.exists(STATIC_AUDIO_DIR)}")

# Import and initialize global audio processor
try:
    from enhanced_audio_processor import EnhancedAudioProcessor, process_audio_for_presentation
    enhanced_audio_processor = EnhancedAudioProcessor()
    print("[INFO] Enhanced audio processor initialized successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import enhanced audio processor: {e}")
    enhanced_audio_processor = None
except Exception as e:
    print(f"[ERROR] Failed to initialize enhanced audio processor: {e}")
    enhanced_audio_processor = None

# Models & Consts
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    # Default to the bundled model relative to this file
    MODEL_PATH = os.path.join(Path(__file__).resolve().parent, "best.pt")
emotion_model = YOLO(MODEL_PATH)

gaze_model = YOLO("yolov8s-pose.pt")

whisper_model = whisper.load_model("turbo", device=DEVICE) 

USE_AZURE = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

if USE_AZURE:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21"
    )
else:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
class PresentationFeedback(BaseModel):
    speechImprovements: str
    speechScore: int
    emotionScore: int
    emotionText: str
    gazeScore: int
    gazeText: str
    movementScore: int
    movementText: str
    shoulderScore: int
    shoulderText: str
    handsScore: int
    gestureText: str
    overallScore: int
    overallSummary: str



mp_fd         = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.1)

HEAD_PAD_TOP, HEAD_PAD_SIDE, HEAD_PAD_BOTTOM = 0.25, 0.25, 0.15
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

mp_pose  = mp.solutions.pose
pose     = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.1)

mp_face  = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    refine_landmarks=False,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

YAW_THR_DEG          = 1.5         # ±15° ⇒ left / right
PITCH_UP_EDGE_DEG    = -25        # ≥ –20° ⇒ up
PITCH_DOWN_EDGE_DEG  = -37        # <  –30° ⇒ down

NOSE, LEYE, REYE = 0, 1, 2   # COCO-pose indices

G_MODEL_POINTS = np.array([
    (0.0,   0.0,   0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3,  32.7, -26.0),
    (-28.9,-28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

LANDMARK_IDS = dict(
    nose_tip=1, chin=199, left_eye_outer=33, right_eye_outer=263,
    mouth_left=61, mouth_right=291
)

# Posture-tracking thresholds (degrees)
SIDE_THR     = 10.0
FORWARD_THR  = 10.0
YAW_THR      = 30

STRAIGHT_THRESHOLD_DEG = 10.0  
GESTURE_RATIO          = 0.15

CONF_THRES        = 0.25                # bbox / kp confidence
KEY_CONF_THR      = 0.10                # accept weaker joints
HIP_GESTURE_R     = 0.15                # original hip-distance ratio
SHOULDER_FACTOR   = 0.50                # hipRatio ≈ 0.5 × shoulderWidth

# COCO key-point indices (YOLO v8 pose uses these)
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_WR,  R_WR  = 9, 10


BATCH              = 1
NUM_WORKERS        = os.cpu_count()
QUEUE_SIZE         = BATCH * 32        

#Per-request State  
frame_index          = 0
class_per_second     = defaultdict(list)
gaze_per_second      = defaultdict(list)
movement_per_second  = defaultdict(list)
shoulder_tilt_per_second = defaultdict(list)   
gesture_per_second       = defaultdict(list)
state_lock           = threading.Lock()

# progress based on analysed frames
processed_frames     = 0
processed_lock       = threading.Lock()

FUN_MESSAGES = [
    "Detecting awkward smiles... yup, that one's forced. 😬",
    "Analyzing eye contact... or lack thereof 👀",
    "Checking if you're making strong points... or just strong gestures 💪",
    "Scanning for power poses... channel your inner TED talk 🧍‍♂️✨",
    "Detecting fidget level: approaching squirrel mode 🐿️",
    "Evaluating if your arms know what they’re doing 🙆",
    "Measuring your confidence by chin height 📏",
    "Is that a dramatic pause or a freeze? 🫠",
    "Posture alert: spine looking suspiciously like a question mark ❓",
    "Analyzing facial expressions... current emotion: existential dread 🫣",
    "Calculating presentation vibes... please wait... ☕",
    "Your body language is currently buffering... 🔄",
    "Optimizing your charisma algorithm... hang tight 🧠",
    "Face detected... now figuring out what it's trying to say 🕵️",
    "Detecting stance: 50% leader, 50% about-to-run 🏃‍♂️💼",
    "Applying motivational filter: 'You got this!' 🌟",
    "Smile check: 1 detected... was that sarcastic? 🤔",
    "Analyzing stage presence... charisma.exe launching 🚀"
]

# Utility helpers  
def reset_state():
    global frame_index, processed_frames
    with state_lock:
        frame_index = 0
        class_per_second.clear()
        gaze_per_second.clear()
        movement_per_second.clear()
        shoulder_tilt_per_second.clear()
        gesture_per_second.clear() 
    with processed_lock:
        processed_frames = 0

def get_random_message(last_change, interval=10):
    now = time.time()
    if now - last_change >= interval:
        return random.choice(FUN_MESSAGES), now
    return None, last_change

def _yaw_pitch_from_keypoints(kxy):
    """
    Return (yaw°, pitch°) using nose & eyes geometry.
    kxy : (17, 2) array in **normalised** coords.
    """
    nx, ny = kxy[NOSE]
    lx, ly = kxy[LEYE]
    rx, ry = kxy[REYE]

    eye_mid_x = 0.5 * (lx + rx)
    eye_mid_y = 0.5 * (ly + ry)
    eye_w     = max(abs(lx - rx), 1e-6)

    yaw   = math.degrees(math.atan2(nx - eye_mid_x, eye_w))
    pitch = math.degrees(math.atan2(eye_mid_y - ny, eye_w))
    return yaw, pitch

def get_direction(yaw, pitch):
    """
      straight : −30° ≤ pitch < −20°
      up       : pitch ≥ −20°
      down     : pitch < −30°
    """
    horiz = "right" if yaw >  YAW_THR_DEG else \
            "left"  if yaw < -YAW_THR_DEG else ""

    if pitch >= PITCH_UP_EDGE_DEG:
        vert = "up"
    elif pitch < PITCH_DOWN_EDGE_DEG:
        vert = "down"
    else:
        vert = ""

    if not horiz and not vert:
        return "straight"
    if not vert:
        return horiz
    if not horiz:
        return vert
    return f"{vert}-{horiz}"

#LLM CALL
def get_feedback_payload(
    dom_emotion,
    dom_gaze:    Dict[int, str],
    move_avg:    Dict[int, float],
    dom_shoulder:Dict[int, str],
    dom_hands:   Dict[int, str],
    segments:    List[Dict[str, Any]]
) -> PresentationFeedback:

    global user
    # Build system + user messages
    system = (
        f"You are a supportive, expert presentation coach helping {user} improve their delivery style. Your response MUST be a **single JSON object** following the exact structure of the PresentationFeedback schema below—no extra commentary, formatting, or explanation.\n\n"
        "...(same content omitted for brevity)..."
    )

    user = f"""
    Below are (1) second‑by‑second analytics extracted from the video and (2) the full speech transcript.

    -----------------------------------------------
    ANALYTICS  (one entry per second)
    • Emotion compiled by transcript speaking time : {dom_emotion}
    • Gaze_sec    : {dom_gaze}
    • Move_avg_sec: {move_avg}
    • Shoulder_sec: {dom_shoulder}
    • Hands_sec   : {dom_hands}
    -----------------------------------------------

    TRANSCRIPT
    {segments}

    Respond with ONLY a JSON object matching the PresentationFeedback schema.
    """

    # Try twice before raising
    for attempt in range(2):
        try:
            completion = client.beta.chat.completions.parse(
                model=os.getenv("OPENAI_MODEL"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                response_format=PresentationFeedback
            )
            feedback: PresentationFeedback = completion.choices[0].message.parsed
            return feedback
        except Exception as e:
            if attempt == 1:
                raise RuntimeError(f"Failed to get PresentationFeedback after 2 attempts: {e}")

#Batch-aware detector functions
def emotion_batch(batch_frames, W, H, batch_secs):
    heads, sec_idx = [], []
    for i, frame in enumerate(batch_frames):
        mp_res = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not mp_res.detections:
            continue
        for det in mp_res.detections:
            bb = det.location_data.relative_bounding_box
            fx1, fy1 = int(bb.xmin * W), int(bb.ymin * H)
            fw,  fh  = int(bb.width * W), int(bb.height * H)
            fx2, fy2 = fx1 + fw, fy1 + fh
            pad_t, pad_s, pad_b = int(fh*HEAD_PAD_TOP), int(fw*HEAD_PAD_SIDE), int(fh*HEAD_PAD_BOTTOM)
            hx1, hy1 = max(0, fx1 - pad_s), max(0, fy1 - pad_t)
            hx2, hy2 = min(W-1, fx2 + pad_s), min(H-1, fy2 + pad_b)
            roi = batch_frames[i][hy1:hy2, hx1:hx2]
            if roi.size == 0:
                continue
            heads.append(roi)
            sec_idx.append(batch_secs[i])

    if not heads:
        return
    results = emotion_model.predict(
        heads, imgsz=640, conf=0.10, device=DEVICE,
        stream=False, verbose=False
    )
    NEUTRAL_CLASSES = {"Fear", "Contempt", "Disgust"}

    for det_res, sec in zip(results, sec_idx):
        preds = det_res.boxes
        if preds is not None and preds.cls.numel() > 0:
            for cls_tensor in preds.cls:
                class_name = emotion_model.names[int(cls_tensor.item())]
                if class_name == "Surprise":
                    class_name = "Happy"
                if class_name in NEUTRAL_CLASSES:
                    class_per_second[sec].append("Neutral")
                else:
                    class_per_second[sec].append(class_name)

def movement_batch(batch_rgbs, batch_secs):
    for img_rgb, sec in zip(batch_rgbs, batch_secs):
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            continue
        lm = res.pose_landmarks.landmark
        try:
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hp = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hp = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        except IndexError:
            continue

        # movement: midpoint of shoulders, normalized [0–1]
        mid_x = (l_sh.x + r_sh.x) / 2.0

        # quantize into 1–10 bins
        # floor(mid_x*10) gives 0–9, +1 gives 1–10, clamp at 10
        bin_idx = int(min(10, math.floor(mid_x * 10) + 1))
        movement_per_second[sec].append(bin_idx)

        #SHOULDER STATE  (straight vs tilted)
        p1 = np.array([l_sh.x, l_sh.y])
        p2 = np.array([r_sh.x, r_sh.y])
        dx, dy   = (p2 - p1)
        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg > 90:   angle_deg -= 180
        if angle_deg < -90:  angle_deg += 180
        label_shoulder = "Shoulders Straight" if abs(angle_deg) <= STRAIGHT_THRESHOLD_DEG else "Shoulders Tilted"
        shoulder_tilt_per_second[sec].append(label_shoulder)


def gesture_batch(batch_rgbs, batch_secs):
    """
    • batch_rgbs : list[np.ndarray]  – RGB frames
    • batch_secs : list[int]        – matching second stamp

    Side-effect:
        gesture_per_second[sec].append("Gesturing" | "Idle Hands")
    """

    global processed_frames  # direct update, no processed_lock

    # One YOLO forward pass for the whole mini-batch
    results = gaze_model(batch_rgbs, conf=CONF_THRES, verbose=False)

    for sec, det in zip(batch_secs, results):
        label_hands = "Idle Hands"          # default

        if det.boxes and det.boxes.shape[0]:
            kxy   = det.keypoints.xyn[0].cpu().numpy()   # (17,2) normalised
            kconf = det.keypoints.conf[0].cpu().numpy()

            if kconf[[L_WR, R_WR]].min() >= KEY_CONF_THR:

                # choose torso anchor: hips if good, else shoulders
                if kconf[[L_HIP, R_HIP]].min() >= KEY_CONF_THR:
                    ref_x = (kxy[L_HIP][0] + kxy[R_HIP][0]) / 2
                    ref_y = (kxy[L_HIP][1] + kxy[R_HIP][1]) / 2
                    thresh = HIP_GESTURE_R
                elif kconf[[L_SHO, R_SHO]].min() >= KEY_CONF_THR:
                    ref_x = (kxy[L_SHO][0] + kxy[R_SHO][0]) / 2
                    ref_y = (kxy[L_SHO][1] + kxy[R_SHO][1]) / 2
                    shoulder_w = max(abs(kxy[L_SHO][0] - kxy[R_SHO][0]), 1e-3)
                    thresh = shoulder_w * SHOULDER_FACTOR
                else:
                    ref_x = ref_y = None   # no stable anchor

                if ref_x is not None:
                    d_l = math.hypot(kxy[L_WR][0] - ref_x,
                                     kxy[L_WR][1] - ref_y)
                    d_r = math.hypot(kxy[R_WR][0] - ref_x,
                                     kxy[R_WR][1] - ref_y)
                    if max(d_l, d_r) > thresh:
                        label_hands = "Gesturing"

        # append label without state_lock
        gesture_per_second[sec].append(label_hands)

def gaze_batch(batch_rgbs, batch_secs, CAM_MAT, DIST, W, H):
    """
    batch_rgbs : list[np.ndarray]  (RGB frames)
    batch_secs : list[int]         (second-timestamp for each frame)

    Side-effect: appends gaze label into gaze_per_second[sec].
    *CAM_MAT, DIST, W, H are unused but kept for interface parity.
    """

    results = gaze_model(batch_rgbs, verbose=False)
    for sec, det in zip(batch_secs, results):
        direction = "straight"  # default if nothing good is detected

        if det.boxes and det.boxes.shape[0]:
            kxy   = det.keypoints.xyn[0].cpu().numpy()   # (17,2)
            kconf = det.keypoints.conf[0].cpu().numpy()
            if kconf[[NOSE, LEYE, REYE]].min() >= 0.20:
                yaw, pitch = _yaw_pitch_from_keypoints(kxy)
                direction  = get_direction(yaw, pitch)

        gaze_per_second[sec].append(direction)

#Flask route 
@app.route('/api/analyze', methods=['POST'])
def analyze():
    global user
    reset_state()
    video = request.files['video']
    user = request.form['userName']
    print(f"Request:{request}")
    print(f"[INFO] Received video: {video.filename} ({video.content_length / 1024:.2f} KB)")
    if not video:
        return {'error': 'No video uploaded'}, 400

    temp_path = os.path.join(tempfile.gettempdir(), video.filename)
    video.save(temp_path)
    print(f"[INFO] Uploaded video: {video.filename} ({os.path.getsize(temp_path) / 1024:.2f} KB)")

    socketio.start_background_task(process_video, temp_path)
    return {'status': 'processing started'}

def emotion_by_segment(dom_emotion, segments):       # NEW
    """Return list whose i-th element is the dominant face-emotion
       during the i-th Whisper segment (or 'None' if no detections)."""
    out = []
    for seg in segments:
        start_sec = int(seg["start"])
        end_sec   = int(math.ceil(seg["end"]))
        window    = list(chain.from_iterable(
                     dom_emotion.get(s, []) for s in range(start_sec, end_sec + 1)))
        if window:
            out.append(Counter(window).most_common(1)[0][0])
        else:
            out.append("None")
    return out

#Core processing 
def process_video(temp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    transcribe_future = executor.submit(
        lambda: whisper_model.transcribe(temp_path)["segments"]
    )

    # unique report ID
    report_id = f"video_{int(time.time())}_{random.randint(1000, 9999)}"
    video_file_tracker[report_id] = temp_path
    print(f"[INFO] Stored video file path for report {report_id}: {temp_path}")
    global frame_index
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    W  = int(640)
    H  = int(480)
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / FPS
    expected_to_process = (total_frames + 7) // 8   # every 8th frame

    focal  = W
    center = (W / 2, H / 2)
    CAM_MAT = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], np.float64)
    DIST = np.zeros((4, 1))

    q        = queue.Queue(maxsize=QUEUE_SIZE)
    SENTINEL = object()

    last_emit_time = time.time()
    emit_interval  = 1.0
    last_msg_time  = time.time()
    current_msg    = random.choice(FUN_MESSAGES)

    # Producer – reads frames and pushes every 8th into queue
    def reader():
        global frame_index
        idx = 0
        nonlocal last_emit_time, last_msg_time, current_msg
        while True:
            ok, frame = cap.read()
            if not ok:
                for _ in range(NUM_WORKERS):
                    q.put(SENTINEL)
                break
            frame = cv2.resize(frame, (640, 480))
            # sample every 8th frame
            if idx % 8 == 0:

                q.put((idx, frame))

            idx += 1
            with state_lock:
                frame_index += 1  

            # fun message throttling
            new_msg, last_msg_time = get_random_message(last_msg_time, 5)
            if new_msg:
                current_msg = new_msg

            time.sleep(0)
    threading.Thread(target=reader, daemon=True).start()

    # Consumer – pulls frames, builds batches, runs detectors
    def consumer():
        batch_frames, batch_secs, batch_rgbs = [], [], []
        while True:
            item = q.get()
            if item is SENTINEL:
                if batch_frames:
                    run_batch(batch_frames, batch_secs, batch_rgbs)
                q.task_done()
                break
            idx, frame = item
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            batch_secs.append(int(idx / FPS))
            batch_rgbs.append(frame_rgb)
            q.task_done()
            if len(batch_frames) == BATCH:
                run_batch(batch_frames, batch_secs, batch_rgbs)
                batch_frames, batch_secs, batch_rgbs = [], [], []

    pool = ThreadPoolExecutor(max_workers=8)

    def run_batch(frames, secs, rgbs):
        nonlocal last_emit_time
        futs = [
            pool.submit(emotion_batch,  frames, W, H, secs),
            pool.submit(gaze_batch,     rgbs, secs, CAM_MAT, DIST, W, H),
            pool.submit(movement_batch, rgbs, secs),
            pool.submit(gesture_batch, rgbs, secs)           
        ]
        wait(futs)

        # update progress based on frames analysed in this batch
        with processed_lock:
            global processed_frames
            processed_frames += len(frames)
            pct_done = int(processed_frames / expected_to_process * 100)

        now = time.time()
        if now - last_emit_time >= emit_interval:
            last_emit_time = now
            socketio.emit('processing-update',
                          {'message': current_msg, 'progress': pct_done})
            socketio.sleep(0)

    workers = [threading.Thread(target=consumer, daemon=True) for _ in range(NUM_WORKERS)]
    for t in workers:
        t.start()
    q.join()
    for t in workers:
        t.join()

    # ── Aggregation / final emit
    dom_emotion = {s: Counter(v).most_common(1)[0][0] for s, v in class_per_second.items()}
    dom_gaze    = {s: Counter(v).most_common(1)[0][0] for s, v in gaze_per_second.items()}
    move_avg    = {s: sum(xs) / len(xs) for s, xs in movement_per_second.items()}
    dom_shoulder= {s: Counter(v).most_common(1)[0][0] for s, v in shoulder_tilt_per_second.items()}
    dom_hands   = {s: Counter(v).most_common(1)[0][0] for s, v in gesture_per_second.items()}

    segments          = transcribe_future.result()
    segment_emotions  = emotion_by_segment(class_per_second, segments)        # NEW
    formatted_segments= "\n".join([f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}"
                                for seg in segments])

    print("Emotion class per second:"); [print(f"  Sec {s}: {c}") for s, c in dom_emotion.items()]
    print("Gaze direction per second:");  [print(f"  Sec {s}: {c}") for s, c in dom_gaze.items()]
    print("Horizontal position per second (0-left, 10-right):")
    [print(f"  Sec {s}: {move_avg[s]:.3f}") for s in sorted(move_avg)]
    print(f"Shoulder tilt per second (> {SIDE_THR}° indicates lean):")
    [print(f"  Sec {s}: {dom_shoulder[s]}°") for s in sorted(dom_shoulder)]
    print(f"Gestures per second (> {FORWARD_THR}° indicates hunch):")
    [print(f"  Sec {s}: {dom_hands[s]}°") for s in sorted(dom_hands)]
    print(formatted_segments)
    feedback = get_feedback_payload(segment_emotions, dom_gaze, move_avg, dom_shoulder, dom_hands, formatted_segments)
    payload = {
        'reportId': report_id,
        'transcriptSegments': formatted_segments,
        'speechImprovements': feedback.speechImprovements,
        'speechScore':    feedback.speechScore,
        'emotion':    dom_emotion,
        'emotionBySegment':    segment_emotions,
        'emotionScore':    feedback.emotionScore,
        'emotionText': feedback.emotionText,
        'gaze':       dom_gaze,
        'gazeScore':    feedback.gazeScore,
        'gazeText': feedback.gazeText,
        'movement': move_avg,
        'movementScore':    feedback.movementScore,
        'movementText': feedback.movementText,
        'shoulder': dom_shoulder,
        'shoulderScore':    feedback.shoulderScore,
        'shoulderText': feedback.shoulderText,
        'gesture': dom_hands,
        'handsScore':    feedback.handsScore,
        'gestureText': feedback.gestureText,
        'overallScore': feedback.overallScore,
        'overallSummary': feedback.overallSummary,
        'videoDuration': duration_seconds
    }
    socketio.emit('processing-complete',
                  {'message': 'Processing done!', 'progress': '100','data': payload})


# @app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    try:
        reset_state()
        
        if 'audio' not in request.files:
            return {'error': 'No audio file found in request'}, 400
            
        audio = request.files['audio']
        
        if audio.filename == '':
            return {'error': 'No audio file selected'}, 400
            
        print(f"[INFO] Received audio: {audio.filename}")
        
        # Create unique temporary path with timestamp
        timestamp = int(time.time())
        filename_base = os.path.splitext(audio.filename)[0]
        temp_filename = f"{filename_base}_{timestamp}.tmp"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        audio.save(temp_path)
        
        if not os.path.exists(temp_path):
            return {'error': 'Failed to save audio file'}, 500
            
        # actual_size = os.path.getsize(temp_path)
        # print(f"[INFO] Audio saved: {actual_size / 1024:.2f} KB")
        
        # if actual_size == 0:
        #     return {'error': 'Audio file is empty'}, 400

        # # Check if enhanced audio processor is available
        # if enhanced_audio_processor is None:
        #     return {'error': 'Audio processor not available. Check server configuration.'}, 500

        # # Use enhanced audio processing
        # socketio.start_background_task(process_audio_for_presentation, temp_path, enhanced_audio_processor, socketio)


        data = get_metrics(temp_path)
        print(data)
        if data:
            return jsonify({
                'status': 'OK',
                'data': data
            }), 200
        if data is False:
            return jsonify({
                'status': 'warning',
                'message': 'No segments recognised' 
            }), 400
        
        return jsonify({
            'status':'error',
            'data': data
        }), 400
    

        # return {'status': 'audio processing started'}

    except Exception as e:
        print(f"[ERROR] analyze_audio endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500



@app.route('/api/transcribe-audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return {'error': 'No audio file found in request'}, 400
            
    audio = request.files['audio']
    
    if audio.filename == '':
        return {'error': 'No audio file selected'}, 400
        
    print(f"[INFO] Received audio: {audio.filename}")
    
    # Create unique temporary path with timestamp
    timestamp = int(time.time())
    filename_base = os.path.splitext(audio.filename)[0]
    temp_filename = f"{filename_base}_{timestamp}.tmp"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    
    audio.save(temp_path)
    
    if not os.path.exists(temp_path):
        return {'error': 'Failed to save audio file'}, 500
    try:
        data = whisper_transcriber(temp_path)
        return jsonify({
            'status': 'OK',
            'data': data,
            'complete_transcript': data['full_text']
        }), 200
    except Exception as e:
        print(f"[ERROR] transcribe_audio endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500






@app.route('/static/generated_audio/<filename>')
def serve_generated_audio(filename):
    """Serve generated audio files with proper headers and debugging"""
    try:
        # Use absolute path to be sure
        audio_dir = os.path.join(os.getcwd(), 'static', 'generated_audio')
        file_path = os.path.join(audio_dir, filename)
        
        print(f"[DEBUG] Looking for audio file: {file_path}")
        print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            # List all files in the directory for debugging
            if os.path.exists(audio_dir):
                files = os.listdir(audio_dir)
                print(f"[DEBUG] Files in audio directory: {files}")
            else:
                print(f"[DEBUG] Audio directory doesn't exist: {audio_dir}")
            
            return jsonify({'error': 'Audio file not found'}), 404
        
        return send_from_directory(
            audio_dir,
            filename,
            as_attachment=False,
            mimetype='audio/mpeg'
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to serve audio file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Audio file serving failed'}), 404

@app.route('/api/health')
def health_check():
    return {'status': 'ok', 'audio_processor': enhanced_audio_processor is not None}


@app.route('/api/debug/audio-files')
def debug_audio_files():
    """Debug endpoint to see what audio files exist"""
    try:
        audio_dir = os.path.join(os.getcwd(), 'static', 'generated_audio')
        if os.path.exists(audio_dir):
            files = os.listdir(audio_dir)
            return jsonify({
                'directory': audio_dir,
                'files': files,
                'count': len(files)
            })
        else:
            return jsonify({'error': 'Audio directory not found', 'directory': audio_dir})
    except Exception as e:
        return jsonify({'error': str(e)})

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

# @app.route('/api/new_audio_analysis', methods=['POST'])
@app.route('/api/analyze-audio', methods=['POST'])
def new_audio_analysis():
    try:
        reset_state()
        
        if 'audio' not in request.files:
            return {'error': 'No audio file found in request'}, 400
            
        audio = request.files['audio']
        
        if audio.filename == '':
            return {'error': 'No audio file selected'}, 400
            
        print(f"[INFO] Received audio for comprehensive analysis: {audio.filename}")
        
        # Create unique temporary path with timestamp
        timestamp = int(time.time())
        filename_base = os.path.splitext(audio.filename)[0]
        original_ext = os.path.splitext(audio.filename)[1] or '.mp3'  # Keep original extension
        temp_filename = f"{filename_base}_{timestamp}{original_ext}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Ensure the temp directory exists
        os.makedirs(tempfile.gettempdir(), exist_ok=True)
        
        audio.save(temp_path)
        
        # Ensure file is completely written (Windows file system delay)
        time.sleep(0.5)
        
        if not os.path.exists(temp_path):
            return {'error': 'Failed to save audio file'}, 500
            
        print(f"[INFO] Audio saved for analysis: {temp_path} ({os.path.getsize(temp_path)} bytes)")
        
        # Start background processing
        socketio.start_background_task(process_comprehensive_audio, temp_path)
        
        return {'status': 'comprehensive audio analysis started'}

    except Exception as e:
        print(f"[ERROR] new_audio_analysis endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

def process_comprehensive_audio(temp_path):
    """Process audio with metrics, transcription, enhancement, vocal dynamics, and TTS generation"""
    try:
        # Normalize path for Windows compatibility
        temp_path = os.path.normpath(temp_path)
        print(f"[INFO] Processing audio file: {temp_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(temp_path):
            socketio.emit('audio-analysis-error', {
                'error': f'Temporary file not found: {temp_path}',
                'stage': 'file_verification'
            })
            return
            
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            socketio.emit('audio-analysis-error', {
                'error': 'Audio file is empty',
                'stage': 'file_verification'
            })
            return
            
        # Debug: Check file format
        try:
            with open(temp_path, 'rb') as f:
                header = f.read(12)
                if header.startswith(b'RIFF'):
                    file_type = "WAV"
                elif header.startswith(b'ID3') or header[0:2] == b'\xff\xfb':
                    file_type = "MP3"
                else:
                    file_type = f"Unknown (header: {header[:4]})"
                print(f"[INFO] Detected file type: {file_type}")
        except Exception as e:
            print(f"[WARNING] Could not detect file type: {e}")
            
        print(f"[INFO] Audio file verified: {file_size} bytes")
        
        # Progress tracking
        socketio.emit('audio-analysis-update', {
            'message': 'Starting comprehensive audio analysis...',
            'progress': 10,
            'stage': 'initialization'
        })
        
        # Step 1: Run metrics, transcription, and vocal dynamics in parallel
        socketio.emit('audio-analysis-update', {
            'message': 'Analyzing speech metrics, vocal dynamics, and transcribing audio...',
            'progress': 20,
            'stage': 'parallel_analysis'
        })
        
        executor = ThreadPoolExecutor(max_workers=8)  # Increased to 3 workers
        
        # Create wrapper functions with better error handling
        def safe_get_metrics(path):
            try:
                print(f"[INFO] Starting metrics analysis for: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Metrics: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Metrics: Audio file is empty")
                
                result = get_metrics(path)
                if result is False:
                    raise ValueError("Metrics: No segments recognized in audio")
                return result
            except Exception as e:
                print(f"[ERROR] Metrics analysis failed: {e}")
                
                # Try manual conversion if it's a format issue
                if "RIFF id" in str(e) or "file does not start with RIFF" in str(e):
                    try:
                        print("[INFO] Attempting manual WAV conversion for metrics...")
                        import subprocess
                        base_path = os.path.splitext(path)[0]
                        converted_path = base_path + "_converted.wav"
                        
                        subprocess.run([
                            "ffmpeg", "-y", "-i", path,
                            "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", 
                            converted_path
                        ], check=True, capture_output=True)
                        
                        if os.path.exists(converted_path):
                            result = get_metrics(converted_path)
                            try:
                                os.remove(converted_path)
                            except:
                                pass
                            return result
                    except Exception as conv_error:
                        print(f"[ERROR] Manual conversion also failed: {conv_error}")
                
                import traceback
                traceback.print_exc()
                raise e
        
        def safe_whisper_transcriber(path):
            try:
                print(f"[INFO] Starting Whisper transcription for: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Whisper: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Whisper: Audio file is empty")
                    
                result = whisper_transcriber(path)
                if not result or not result.get('full_text'):
                    raise ValueError("Whisper: Failed to transcribe audio or no text detected")
                return result
            except Exception as e:
                print(f"[ERROR] Whisper transcription failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        def safe_vocal_dynamics_analyzer(path):
            try:
                print(f"[INFO] Starting vocal dynamics analysis for: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Vocal Dynamics: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Vocal Dynamics: Audio file is empty")
                    
                result = analyze_vocal_dynamics(path)
                if not result:
                    raise ValueError("Vocal Dynamics: Analysis failed")
                return result
            except Exception as e:
                print(f"[ERROR] Vocal dynamics analysis failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Submit all three tasks with the safe wrappers
        metrics_future = executor.submit(safe_get_metrics, temp_path)
        transcription_future = executor.submit(safe_whisper_transcriber, temp_path)
        vocal_dynamics_future = executor.submit(safe_vocal_dynamics_analyzer, temp_path)
        
        # Wait for metrics
        socketio.emit('audio-analysis-update', {
            'message': 'Processing speech quality metrics...',
            'progress': 30,
            'stage': 'metrics'
        })
        
        metrics_error = None
        try:
            metrics_data = metrics_future.result(timeout=60)
        except Exception as e:
            print(f"[ERROR] Metrics analysis failed: {e}")
            metrics_data = None
            metrics_error = str(e)
        
        # Wait for vocal dynamics
        socketio.emit('audio-analysis-update', {
            'message': 'Analyzing vocal dynamics and presentation style...',
            'progress': 40,
            'stage': 'vocal_dynamics'
        })
        
        vocal_dynamics_error = None
        try:
            vocal_dynamics_data = vocal_dynamics_future.result(timeout=90)
        except Exception as e:
            print(f"[ERROR] Vocal dynamics analysis failed: {e}")
            vocal_dynamics_data = None
            vocal_dynamics_error = str(e)
        
        # Wait for transcription
        socketio.emit('audio-analysis-update', {
            'message': 'Completing transcription and gender detection...',
            'progress': 55,
            'stage': 'transcription'
        })
        
        transcription_error = None
        try:
            transcription_data = transcription_future.result(timeout=120)
        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            transcription_data = None
            transcription_error = str(e)
        
        # Check if we have essential data to continue
        if not transcription_data or not transcription_data.get('full_text'):
            error_data = {
                'error': 'Failed to transcribe audio. Cannot proceed with analysis.',
                'stage': 'transcription',
                'details': transcription_error,
                'partial_results': {
                    'metrics_success': metrics_data is not None,
                    'metrics_error': metrics_error,
                    'vocal_dynamics_success': vocal_dynamics_data is not None,
                    'vocal_dynamics_error': vocal_dynamics_error,
                    'transcription_success': False,
                    'transcription_error': transcription_error
                }
            }
            error_data = convert_numpy_types(error_data)
            socketio.emit('audio-analysis-error', error_data)
            return
        
        # Step 2: Enhance transcript
        socketio.emit('audio-analysis-update', {
            'message': 'Enhancing transcript for optimal presentation...',
            'progress': 65,
            'stage': 'enhancement'
        })
        
        enhancement_error = None
        try:
            enhanced_data = enhance_transcript_for_presentation(transcription_data['full_text'])
        except Exception as e:
            print(f"[ERROR] Transcript enhancement failed: {e}")
            enhancement_error = str(e)
            # Fallback to original transcript
            enhanced_data = {
                "enhanced_text": transcription_data['full_text'],
                "improvements_made": ["Enhancement failed - using original transcript"],
                "presentation_tips": ["Practice with the original transcript"]
            }
        
        # Step 3: Generate TTS audio
        socketio.emit('audio-analysis-update', {
            'message': 'Generating optimized speech audio...',
            'progress': 80,
            'stage': 'tts_generation'
        })
        
        # Use detected gender from transcription, fallback to Female
        detected_gender = transcription_data.get('speaker_gender', 'unknown')
        if detected_gender.lower() == 'male':
            tts_gender = 'Male'
        elif detected_gender.lower() == 'female':
            tts_gender = 'Female'
        else:
            tts_gender = 'Female'  # Default fallback
        
        try:
            # Generate unique filename for the enhanced audio
            timestamp = int(time.time())
            enhanced_filename = f"enhanced_speech_{timestamp}"
            
            success, audio_result = get_speech(
                text=enhanced_data['enhanced_text'],
                gender=tts_gender,
                filename=enhanced_filename
            )
            
            if not success:
                print(f"[ERROR] TTS generation failed: {audio_result}")
                audio_file_path = None
                tts_error = audio_result
            else:
                audio_file_path = audio_result
                tts_error = None
                
        except Exception as e:
            print(f"[ERROR] TTS generation exception: {e}")
            audio_file_path = None
            tts_error = str(e)
        
        # Step 4: Compile comprehensive results
        socketio.emit('audio-analysis-update', {
            'message': 'Compiling advanced analysis results...',
            'progress': 95,
            'stage': 'compilation'
        })
        
        # Calculate scores for presentation metrics
        original_word_count = len(transcription_data['full_text'].split()) if transcription_data['full_text'] else 0
        enhanced_word_count = len(enhanced_data['enhanced_text'].split()) if enhanced_data['enhanced_text'] else 0

        # Count filler words in original (rough estimate)
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'so', 'well']
        original_lower = transcription_data['full_text'].lower() if transcription_data['full_text'] else ""
        filler_count = sum(original_lower.count(filler) for filler in filler_words)

        # Enhanced scoring with vocal dynamics
        clarity_score = metrics_data.get('accuracy', 75) if metrics_data else 75
        pace_score = metrics_data.get('fluency', 70) if metrics_data else 70
        confidence_score = max(0, 100 - (filler_count * 5))
        engagement_score = metrics_data.get('prosody', 80) if metrics_data else 80

        # Add vocal dynamics influence to scores
        if vocal_dynamics_data:
            # Boost clarity score with vocal clarity
            vocal_clarity_boost = vocal_dynamics_data.get('vocal_health_indicators', {}).get('vocal_clarity', 0)
            clarity_score = min(100, clarity_score + (vocal_clarity_boost - 50) * 0.3)
            
            # Boost engagement with pitch variation
            pitch_variation_boost = vocal_dynamics_data.get('pitch_dynamics', {}).get('variation_score', 0)
            engagement_score = min(100, engagement_score + (pitch_variation_boost - 50) * 0.4)
            
            # Adjust pace score with rhythm analysis
            rhythm_boost = vocal_dynamics_data.get('rhythm_analysis', {}).get('rhythm_score', 0)
            pace_score = min(100, (pace_score + rhythm_boost) / 2)

        # Calculate overall score as average
        overall_score = int((clarity_score + pace_score + confidence_score + engagement_score) / 4)
        
        # Build comprehensive payload to match frontend expectations
        comprehensive_payload = {
            # Basic transcript data (matches frontend expectations)
            'transcriptSegments': transcription_data['full_text'],
            'enhancedTranscript': enhanced_data['enhanced_text'],
            'enhancedAudioUrl': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'duration': float(transcription_data.get('duration', 0)),
            'language': transcription_data.get('language', 'en'),
            'overallScore': overall_score,
            
            # Presentation metrics (enhanced with vocal dynamics)
            'presentationMetrics': {
                'clarity_score': int(clarity_score),
                'pace_score': int(pace_score),
                'confidence_score': int(confidence_score),
                'engagement_score': int(engagement_score),
                'clarity_feedback': f"Your pronunciation accuracy is {clarity_score:.0f}%. {get_vocal_clarity_feedback(vocal_dynamics_data, clarity_score)}",
                'pace_feedback': f"Your speaking rhythm scores {pace_score:.0f}%. {get_pace_feedback(vocal_dynamics_data, pace_score)}",
                'confidence_feedback': f"Detected {filler_count} filler words. {get_confidence_feedback(vocal_dynamics_data, confidence_score)}",
                'engagement_feedback': f"Your vocal engagement scores {engagement_score:.0f}%. {get_engagement_feedback(vocal_dynamics_data, engagement_score)}"
            },
            
            # NEW: Advanced Vocal Dynamics Section
            'vocalDynamics': vocal_dynamics_data if vocal_dynamics_data else {
                'error': 'Vocal dynamics analysis failed',
                'pitch_dynamics': {'variation_score': 0, 'dynamic_range': 'Analysis Failed'},
                'volume_dynamics': {'energy_score': 0, 'dynamic_presence': 'Unknown'},
                'rhythm_analysis': {'rhythm_score': 0, 'pace_category': 'Unknown'},
                'pause_analysis': {'strategic_score': 0, 'pause_effectiveness': 'Unknown'},
                'overall_dynamics_score': 0,
                'presentation_readiness': {'readiness_level': 'Analysis Failed'}
            },
            
            # Enhancement data (matches frontend structure)
            'enhancement': {
                'key_changes': enhanced_data.get('improvements_made', []),
                'speaking_tips': enhanced_data.get('presentation_tips', []) + get_vocal_dynamics_tips(vocal_dynamics_data),
                'summary': f"Your speech has been enhanced by removing filler words and improving flow. Word count changed from {original_word_count} to {enhanced_word_count} words."
            },
            
            # Speech analysis data (enhanced with vocal metrics)
            'speechAnalysis': {
                'word_count': original_word_count,
                'speaking_rate': int((original_word_count / transcription_data.get('duration', 1)) * 60) if transcription_data.get('duration', 0) > 0 else 0,
                'pace_feedback': f"Speaking rate: {int((original_word_count / transcription_data.get('duration', 1)) * 60)} words per minute. {get_detailed_pace_feedback(vocal_dynamics_data)}",
                'filler_feedback': f"Detected {filler_count} filler words. Focus on pausing instead of using 'um', 'uh', 'like', etc.",
                'vocal_summary': get_vocal_summary(vocal_dynamics_data)
            },
            
            # Technical metadata (for debugging/additional info)
            'technicalData': {
                'transcription_metadata': {
                    'language': transcription_data.get('language', 'unknown'),
                    'duration': float(transcription_data.get('duration', 0)),
                    'speaker_gender': transcription_data.get('speaker_gender', 'unknown'),
                    'pitch_hz': float(transcription_data.get('pitch_hz', 0)),
                    'segments': transcription_data.get('segments', [])
                },
                'speech_metrics': metrics_data if metrics_data else {
                    'error': 'Metrics analysis failed',
                    'pronunciation': 0,
                    'accuracy': 0,
                    'fluency': 0,
                    'prosody': 0
                },
                'vocal_dynamics_raw': vocal_dynamics_data,
                'generated_audio': {
                    'file_path': audio_file_path,
                    'voice_gender': tts_gender,
                    'generation_success': audio_file_path is not None,
                    'error': tts_error
                },
                'component_status': {
                    'metrics': {
                        'success': metrics_data is not None,
                        'error': metrics_error
                    },
                    'vocal_dynamics': {
                        'success': vocal_dynamics_data is not None,
                        'error': vocal_dynamics_error
                    },
                    'transcription': {
                        'success': transcription_data is not None,
                        'error': transcription_error
                    },
                    'enhancement': {
                        'success': enhancement_error is None,
                        'error': enhancement_error
                    },
                    'tts_generation': {
                        'success': audio_file_path is not None,
                        'error': tts_error
                    }
                },
                'analysis_metadata': {
                    'original_word_count': original_word_count,
                    'enhanced_word_count': enhanced_word_count,
                    'estimated_filler_count': filler_count,
                    'word_reduction': original_word_count - enhanced_word_count,
                    'processing_timestamp': timestamp,
                    'has_metrics': metrics_data is not None,
                    'has_vocal_dynamics': vocal_dynamics_data is not None,
                    'has_enhancement': enhanced_data is not None,
                    'has_generated_audio': audio_file_path is not None,
                    'overall_success': all([
                        transcription_data is not None,
                    ])
                }
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        comprehensive_payload = convert_numpy_types(comprehensive_payload)
        
        # Final success emission with enhanced summary
        summary_data = {
            'components_completed': sum([
                metrics_data is not None,
                vocal_dynamics_data is not None,
                transcription_data is not None,
                enhancement_error is None,
                audio_file_path is not None
            ]),
            'total_components': 5,  # Updated to 5 components
            'audio_file_location': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'analysis_success': True,
            'overall_score': comprehensive_payload['overallScore'],
            'key_metrics': {
                'word_count': original_word_count,
                'duration': float(transcription_data.get('duration', 0)),
                'filler_count': filler_count,
                'enhancement_available': audio_file_path is not None,
                'vocal_dynamics_score': vocal_dynamics_data.get('overall_dynamics_score', 0) if vocal_dynamics_data else 0,
                'presentation_readiness': vocal_dynamics_data.get('presentation_readiness', {}).get('readiness_level', 'Unknown') if vocal_dynamics_data else 'Unknown'
            }
        }
        
        # Convert summary data as well
        summary_data = convert_numpy_types(summary_data)
        
        socketio.emit('audio-analysis-complete', {
            'message': 'Advanced vocal analysis complete!',
            'progress': 100,
            'stage': 'complete',
            'data': comprehensive_payload,
            'summary': summary_data
        })
        
        print("[INFO] Advanced vocal dynamics analysis completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Comprehensive audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_data = {
            'error': f'Processing failed: {str(e)}',
            'stage': 'processing'
        }
        error_data = convert_numpy_types(error_data)
        socketio.emit('audio-analysis-error', error_data)
    
    finally:
        # Cleanup temporary file
        try:
            time.sleep(1)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[INFO] Cleaned up temporary file: {temp_path}")
        except PermissionError:
            time.sleep(2)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[INFO] Cleaned up temporary file (delayed): {temp_path}")
            except Exception as e:
                print(f"[WARNING] Could not cleanup temp file: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp file: {e}")

# Helper functions for generating vocal dynamics feedback
def get_vocal_clarity_feedback(vocal_dynamics_data, clarity_score):
    """Generate specific feedback based on vocal clarity analysis"""
    if not vocal_dynamics_data:
        return "Focus on clear articulation and proper breathing."
    
    vocal_health = vocal_dynamics_data.get('vocal_health_indicators', {})
    clarity = vocal_health.get('vocal_clarity', 50)
    breathiness = vocal_health.get('breathiness_level', 50)
    
    if clarity > 80:
        return "Excellent vocal clarity! Your voice projects well."
    elif clarity > 60:
        return "Good vocal clarity. Consider warming up your voice before speaking."
    else:
        feedback = "Work on vocal clarity through breathing exercises."
        if breathiness > 70:
            feedback += " Reduce breathiness by supporting your voice with your diaphragm."
        return feedback

def get_pace_feedback(vocal_dynamics_data, pace_score):
    """Generate specific feedback based on rhythm analysis"""
    if not vocal_dynamics_data:
        return "Focus on maintaining a steady, comfortable pace."
    
    rhythm = vocal_dynamics_data.get('rhythm_analysis', {})
    pace_category = rhythm.get('pace_category', 'Unknown')
    rhythm_score = rhythm.get('rhythm_score', 50)
    
    if pace_category == 'Very Fast':
        return "Slow down! Your pace is too fast for clear comprehension."
    elif pace_category == 'Fast':
        return "Consider slowing down slightly for better audience engagement."
    elif pace_category == 'Optimal':
        return "Excellent pacing! You're speaking at an ideal rate."
    elif pace_category == 'Slow':
        return "Consider picking up the pace slightly to maintain energy."
    elif pace_category == 'Very Slow':
        return "Increase your speaking pace to keep audience attention."
    else:
        return f"Your rhythm consistency scores {rhythm_score}/100. Work on maintaining steady tempo."

def get_confidence_feedback(vocal_dynamics_data, confidence_score):
    """Generate confidence feedback based on vocal analysis"""
    if not vocal_dynamics_data:
        return "Practice reducing filler words and speaking with conviction."
    
    pitch_dynamics = vocal_dynamics_data.get('pitch_dynamics', {})
    monotone_risk = pitch_dynamics.get('monotone_risk', 50)
    
    feedback = "Practice reducing filler words and use strategic pauses instead."
    
    if monotone_risk > 70:
        feedback += " Add more pitch variation to sound more engaging and confident."
    elif monotone_risk < 30:
        feedback += " Your vocal variety shows confidence - maintain this energy!"
    
    return feedback

def get_engagement_feedback(vocal_dynamics_data, engagement_score):
    """Generate engagement feedback based on vocal dynamics"""
    if not vocal_dynamics_data:
        return "Add more energy and enthusiasm to capture audience attention."
    
    pitch_dynamics = vocal_dynamics_data.get('pitch_dynamics', {})
    volume_dynamics = vocal_dynamics_data.get('volume_dynamics', {})
    
    variation_score = pitch_dynamics.get('variation_score', 50)
    energy_score = volume_dynamics.get('energy_score', 50)
    dynamic_presence = volume_dynamics.get('dynamic_presence', 'Moderate')
    
    if variation_score > 80 and energy_score > 70:
        return "Excellent vocal engagement! You use pitch and volume dynamically."
    elif variation_score < 40:
        return "Add more pitch variation to keep listeners engaged."
    elif energy_score < 40:
        return "Increase your vocal energy and volume dynamics."
    else:
        return f"Good vocal presence ({dynamic_presence}). Work on consistency throughout your speech."

def get_vocal_dynamics_tips(vocal_dynamics_data):
    """Generate additional speaking tips based on vocal dynamics analysis"""
    if not vocal_dynamics_data:
        return ["Practice vocal warm-ups before speaking", "Record yourself to monitor improvement"]
    
    tips = []
    
    # Pitch-based tips
    pitch_dynamics = vocal_dynamics_data.get('pitch_dynamics', {})
    if pitch_dynamics.get('monotone_risk', 0) > 60:
        tips.append("Practice reading with exaggerated expression to develop pitch variation")
    
    # Volume-based tips
    volume_dynamics = vocal_dynamics_data.get('volume_dynamics', {})
    if volume_dynamics.get('energy_score', 0) < 50:
        tips.append("Work on breath support to maintain vocal energy throughout your speech")
    
    # Pause-based tips
    pause_analysis = vocal_dynamics_data.get('pause_analysis', {})
    if pause_analysis.get('strategic_score', 0) < 60:
        tips.append("Practice using strategic pauses for emphasis and to give audience time to process")
    
    # Rhythm-based tips
    rhythm_analysis = vocal_dynamics_data.get('rhythm_analysis', {})
    if rhythm_analysis.get('rhythm_score', 0) < 60:
        tips.append("Use a metronome or backing track to practice consistent speaking rhythm")
    
    return tips

def get_detailed_pace_feedback(vocal_dynamics_data):
    """Get detailed pace feedback including rhythm analysis"""
    if not vocal_dynamics_data:
        return "Ideal range is 140-160 WPM."
    
    rhythm = vocal_dynamics_data.get('rhythm_analysis', {})
    pace_category = rhythm.get('pace_category', 'Unknown')
    estimated_pace = rhythm.get('estimated_pace_wpm', 0)
    
    feedback = f"Ideal range is 140-160 WPM. Your estimated pace: {estimated_pace} WPM ({pace_category})."
    
    if estimated_pace > 180:
        feedback += " Consider slowing down for better comprehension."
    elif estimated_pace < 120:
        feedback += " Consider increasing pace to maintain audience engagement."
    
    return feedback

def get_vocal_summary(vocal_dynamics_data):
    """Generate a comprehensive vocal summary"""
    if not vocal_dynamics_data:
        return "Complete vocal analysis not available."
    
    readiness = vocal_dynamics_data.get('presentation_readiness', {})
    overall_dynamics = vocal_dynamics_data.get('overall_dynamics_score', 0)
    readiness_level = readiness.get('readiness_level', 'Unknown')
    
    return f"Vocal Dynamics Score: {overall_dynamics}/100. Presentation Readiness: {readiness_level}. Your voice shows {vocal_dynamics_data.get('pitch_dynamics', {}).get('dynamic_range', 'moderate variation')} and {vocal_dynamics_data.get('volume_dynamics', {}).get('dynamic_presence', 'moderate presence')}."



@app.route('/api/extract-audio-and-analyze', methods=['POST'])
def extract_audio_and_analyze():
    """
    Extract audio from video used in video analysis and run comprehensive audio analysis
    """
    try:
        data = request.get_json()
        report_id = data.get('reportId')
        video_report_id = data.get('videoReportId')
        
        print(f"[INFO] Audio analysis request for report: {report_id}, video report: {video_report_id}")
        
        # Try to find the video file path from our tracker
        video_path = None
        
        # Look for the video file in our tracker
        for tracked_id, path in video_file_tracker.items():
            if str(video_report_id) in tracked_id or tracked_id == str(video_report_id):
                video_path = path
                break
        
        # If not found in tracker, check if there are any recent video files
        if not video_path:
            # Get the most recent video file from temp directory as fallback
            temp_dir = tempfile.gettempdir()
            video_files = []
            
            for file in os.listdir(temp_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(temp_dir, file)
                    # Check if file was modified in the last hour
                    if os.path.getmtime(file_path) > time.time() - 3600:
                        video_files.append((file_path, os.path.getmtime(file_path)))
            
            if video_files:
                # Use the most recently modified video file
                video_path = max(video_files, key=lambda x: x[1])[0]
                print(f"[INFO] Using most recent video file as fallback: {video_path}")
        
        if not video_path or not os.path.exists(video_path):
            return {'error': 'Video file not found. The video may have been cleaned up.'}, 400
            
        print(f"[INFO] Extracting audio from video: {video_path}")
        
        # Create unique audio filename
        timestamp = int(time.time())
        audio_filename = f"extracted_audio_{report_id}_{timestamp}.wav"
        audio_temp_path = os.path.join(tempfile.gettempdir(), audio_filename)
        
        # Check if FFmpeg is available
        def check_ffmpeg():
            try:
                result = subprocess.run(['ffmpeg', '-version'], 
                                      capture_output=True, text=True, timeout=10)
                return result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                return False
        
        print("[INFO] Checking FFmpeg availability...")
        if not check_ffmpeg():
            error_msg = ("FFmpeg not found or not working. Please ensure FFmpeg is installed and added to your system PATH. "
                        "You can download it from https://ffmpeg.org/download.html")
            print(f"[ERROR] {error_msg}")
            return {'error': error_msg}, 500
        
        print("[INFO] FFmpeg is available, proceeding with extraction...")
        
        # Extract audio from video using ffmpeg with improved error handling
        try:
            print(f"[INFO] Starting audio extraction to: {audio_temp_path}")
            
            # FFmpeg command with better parameters for stability on Windows
            ffmpeg_cmd = [
                'ffmpeg', 
                '-nostdin',  # Don't wait for input
                '-y',        # Overwrite output files
                '-hide_banner',  # Reduce output verbosity
                '-loglevel', 'error',  # Only show errors
                '-i', video_path,  # Input video
                '-vn',       # No video
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz sample rate
                '-f', 'wav', # Force WAV format
                audio_temp_path
            ]
            
            print(f"[DEBUG] FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Use Popen for better control on Windows
            print("[INFO] Starting FFmpeg process...")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,  # Provide stdin pipe
                text=True,
                cwd=tempfile.gettempdir(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Close stdin immediately to prevent hanging
            process.stdin.close()
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=120)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                print("[ERROR] FFmpeg process timed out, terminating...")
                process.kill()
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(ffmpeg_cmd, 120)
            
            # Create result-like object for compatibility
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = Result(return_code, stdout, stderr)
            print(f"[DEBUG] FFmpeg completed with return code: {return_code}")
            if stdout:
                print(f"[DEBUG] FFmpeg stdout: {stdout[:500]}...")  # Limit output
            if stderr:
                print(f"[DEBUG] FFmpeg stderr: {stderr[:500]}...")  # Limit output
            
            # Check if FFmpeg succeeded
            if result.returncode != 0:
                print(f"[ERROR] FFmpeg failed with return code {result.returncode}")
                print(f"[ERROR] FFmpeg stderr: {result.stderr}")
                print(f"[ERROR] FFmpeg stdout: {result.stdout}")
                return {'error': f'Audio extraction failed: {result.stderr}'}, 500
            
            # Verify the output file was created and has content
            if not os.path.exists(audio_temp_path):
                print(f"[ERROR] Audio file was not created: {audio_temp_path}")
                return {'error': 'Audio extraction failed: output file not created'}, 500
                
            file_size = os.path.getsize(audio_temp_path)
            if file_size == 0:
                print(f"[ERROR] Audio file is empty: {audio_temp_path}")
                return {'error': 'Audio extraction failed: output file is empty'}, 500
                
            print(f"[INFO] Audio extracted successfully: {audio_temp_path} ({file_size} bytes)")
            
        except subprocess.TimeoutExpired:
            print("[ERROR] FFmpeg timed out during audio extraction")
            # Try alternative approach with simpler command
            print("[INFO] Attempting alternative extraction method...")
            return try_alternative_extraction(video_path, audio_temp_path, report_id)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg process error: {e}")
            return try_alternative_extraction(video_path, audio_temp_path, report_id)
        except FileNotFoundError:
            print("[ERROR] FFmpeg executable not found")
            return {'error': 'FFmpeg not found. Please install FFmpeg and add it to your system PATH.'}, 500
        except Exception as e:
            print(f"[ERROR] Unexpected error during audio extraction: {e}")
            import traceback
            traceback.print_exc()
            return try_alternative_extraction(video_path, audio_temp_path, report_id)
        
        # Start comprehensive audio analysis in background
        print(f"[INFO] Starting background audio analysis for report {report_id}")
        socketio.start_background_task(process_comprehensive_audio_for_video, audio_temp_path, report_id)
        
        return {
            'status': 'success',
            'message': 'Audio extraction completed, analysis in progress',
            'reportId': report_id
        }

    except Exception as e:
        print(f"[ERROR] extract_audio_and_analyze endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500


def try_alternative_extraction(video_path, audio_temp_path, report_id):
    """Try alternative methods for audio extraction when FFmpeg fails"""
    print("[INFO] Trying alternative audio extraction methods...")
    
    # Method 1: Try simpler FFmpeg command
    try:
        print("[INFO] Attempting simpler FFmpeg command...")
        simple_cmd = [
            'ffmpeg', '-y', '-i', video_path, 
            '-vn', '-ar', '16000', '-ac', '1', 
            audio_temp_path
        ]
        
        result = subprocess.run(
            simple_cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            input='',  # Send empty input to stdin
            cwd=tempfile.gettempdir()
        )
        
        if result.returncode == 0 and os.path.exists(audio_temp_path) and os.path.getsize(audio_temp_path) > 0:
            print("[INFO] Simple FFmpeg command succeeded!")
            socketio.start_background_task(process_comprehensive_audio_for_video, audio_temp_path, report_id)
            return {
                'status': 'success',
                'message': 'Audio extraction completed (alternative method), analysis in progress',
                'reportId': report_id
            }
    except Exception as e:
        print(f"[WARNING] Simple FFmpeg also failed: {e}")
    
    # Method 2: Try MoviePy as fallback
    try:
        print("[INFO] Attempting MoviePy extraction...")
        from moviepy.editor import VideoFileClip
        
        # Load video and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Write audio file
        audio.write_audiofile(
            audio_temp_path, 
            fps=16000,
            nbytes=2,
            codec='pcm_s16le',
            logger=None,
            verbose=False
        )
        
        # Clean up
        audio.close()
        video.close()
        
        if os.path.exists(audio_temp_path) and os.path.getsize(audio_temp_path) > 0:
            print("[INFO] MoviePy extraction succeeded!")
            socketio.start_background_task(process_comprehensive_audio_for_video, audio_temp_path, report_id)
            return {
                'status': 'success',
                'message': 'Audio extraction completed (MoviePy method), analysis in progress',
                'reportId': report_id
            }
        else:
            raise Exception("MoviePy did not create audio file")
            
    except ImportError:
        print("[WARNING] MoviePy not available (pip install moviepy)")
    except Exception as e:
        print(f"[WARNING] MoviePy extraction failed: {e}")
    
    # Method 3: Last resort - try basic system call
    try:
        print("[INFO] Attempting basic system command...")
        cmd = f'ffmpeg -y -i "{video_path}" -vn -ar 16000 -ac 1 "{audio_temp_path}"'
        
        exit_code = os.system(cmd)
        if exit_code == 0 and os.path.exists(audio_temp_path) and os.path.getsize(audio_temp_path) > 0:
            print("[INFO] System command succeeded!")
            socketio.start_background_task(process_comprehensive_audio_for_video, audio_temp_path, report_id)
            return {
                'status': 'success',
                'message': 'Audio extraction completed (system method), analysis in progress',
                'reportId': report_id
            }
    except Exception as e:
        print(f"[WARNING] System command failed: {e}")
    
    # All methods failed
    return {
        'error': ('Audio extraction failed with all methods. Please ensure:\n'
                 '1. FFmpeg is properly installed and in PATH\n'
                 '2. The video file is not corrupted\n'
                 '3. You have write permissions to temp directory\n'
                 '4. Consider installing MoviePy: pip install moviepy')
    }, 500

def process_comprehensive_audio_for_video(temp_path, report_id):
    """
    Process audio extracted from video with comprehensive analysis
    Emits results specifically for video-to-audio analysis
    """
    try:
                # Progress tracking
        socketio.emit('video-audio-analysis-update', {
            'message': 'Starting comprehensive audio analysis from video...',
            'progress': 10,
            'stage': 'initialization',
            'reportId': report_id
        })
        # Normalize path for Windows compatibility
        temp_path = os.path.normpath(temp_path)
        print(f"[INFO] Processing extracted audio file: {temp_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(temp_path):
            socketio.emit('video-audio-analysis-error', {
                'error': f'Extracted audio file not found: {temp_path}',
                'stage': 'file_verification',
                'reportId': report_id
            })
            return
            
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            socketio.emit('video-audio-analysis-error', {
                'error': 'Extracted audio file is empty',
                'stage': 'file_verification',
                'reportId': report_id
            })
            return
            
        print(f"[INFO] Extracted audio file verified: {file_size} bytes")
        

        
        # Step 1: Run metrics, transcription, and vocal dynamics in parallel
        socketio.emit('video-audio-analysis-update', {
            'message': 'Analyzing speech metrics, vocal dynamics, and transcribing audio...',
            'progress': 20,
            'stage': 'parallel_analysis',
            'reportId': report_id
        })
        
        executor = ThreadPoolExecutor(max_workers=3)
        
        # Create wrapper functions with better error handling
        def safe_get_metrics(path):
            try:
                print(f"[INFO] Starting metrics analysis for extracted audio: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Metrics: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Metrics: Audio file is empty")
                
                result = get_metrics(path)
                if result is False:
                    raise ValueError("Metrics: No segments recognized in audio")
                return result
            except Exception as e:
                print(f"[ERROR] Metrics analysis failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        def safe_whisper_transcriber(path):
            try:
                print(f"[INFO] Starting Whisper transcription for extracted audio: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Whisper: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Whisper: Audio file is empty")
                    
                result = whisper_transcriber(path)
                if not result or not result.get('full_text'):
                    raise ValueError("Whisper: Failed to transcribe audio or no text detected")
                return result
            except Exception as e:
                print(f"[ERROR] Whisper transcription failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        def safe_vocal_dynamics_analyzer(path):
            try:
                print(f"[INFO] Starting vocal dynamics analysis for extracted audio: {path}")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Vocal Dynamics: Audio file not found: {path}")
                if os.path.getsize(path) == 0:
                    raise ValueError("Vocal Dynamics: Audio file is empty")
                    
                result = analyze_vocal_dynamics(path)
                if not result:
                    raise ValueError("Vocal Dynamics: Analysis failed")
                return result
            except Exception as e:
                print(f"[ERROR] Vocal dynamics analysis failed: {e}")
                import traceback
                traceback.print_exc()
                raise e
        
        # Submit all three tasks with the safe wrappers
        metrics_future = executor.submit(safe_get_metrics, temp_path)
        transcription_future = executor.submit(safe_whisper_transcriber, temp_path)
        vocal_dynamics_future = executor.submit(safe_vocal_dynamics_analyzer, temp_path)
        
        # Wait for metrics
        socketio.emit('video-audio-analysis-update', {
            'message': 'Processing speech quality metrics...',
            'progress': 30,
            'stage': 'metrics',
            'reportId': report_id
        })
        
        metrics_error = None
        try:
            metrics_data = metrics_future.result(timeout=60)
        except Exception as e:
            print(f"[ERROR] Metrics analysis failed: {e}")
            metrics_data = None
            metrics_error = str(e)
        
        # Wait for vocal dynamics
        socketio.emit('video-audio-analysis-update', {
            'message': 'Analyzing vocal dynamics and presentation style...',
            'progress': 40,
            'stage': 'vocal_dynamics',
            'reportId': report_id
        })
        
        vocal_dynamics_error = None
        try:
            vocal_dynamics_data = vocal_dynamics_future.result(timeout=90)
        except Exception as e:
            print(f"[ERROR] Vocal dynamics analysis failed: {e}")
            vocal_dynamics_data = None
            vocal_dynamics_error = str(e)
        
        # Wait for transcription
        socketio.emit('video-audio-analysis-update', {
            'message': 'Completing transcription and gender detection...',
            'progress': 55,
            'stage': 'transcription',
            'reportId': report_id
        })
        
        transcription_error = None
        try:
            transcription_data = transcription_future.result(timeout=120)
        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            transcription_data = None
            transcription_error = str(e)
        
        # Check if we have essential data to continue
        if not transcription_data or not transcription_data.get('full_text'):
            error_data = {
                'error': 'Failed to transcribe extracted audio. Cannot proceed with analysis.',
                'stage': 'transcription',
                'reportId': report_id,
                'details': transcription_error,
                'partial_results': {
                    'metrics_success': metrics_data is not None,
                    'metrics_error': metrics_error,
                    'vocal_dynamics_success': vocal_dynamics_data is not None,
                    'vocal_dynamics_error': vocal_dynamics_error,
                    'transcription_success': False,
                    'transcription_error': transcription_error
                }
            }
            error_data = convert_numpy_types(error_data)
            socketio.emit('video-audio-analysis-error', error_data)
            return
        
        # Step 2: Enhance transcript
        socketio.emit('video-audio-analysis-update', {
            'message': 'Enhancing transcript for optimal presentation...',
            'progress': 65,
            'stage': 'enhancement',
            'reportId': report_id
        })
        
        enhancement_error = None
        try:
            enhanced_data = enhance_transcript_for_presentation(transcription_data['full_text'])
        except Exception as e:
            print(f"[ERROR] Transcript enhancement failed: {e}")
            enhancement_error = str(e)
            # Fallback to original transcript
            enhanced_data = {
                "enhanced_text": transcription_data['full_text'],
                "improvements_made": ["Enhancement failed - using original transcript"],
                "presentation_tips": ["Practice with the original transcript"]
            }
        
        # Step 3: Generate TTS audio
        socketio.emit('video-audio-analysis-update', {
            'message': 'Generating optimized speech audio...',
            'progress': 80,
            'stage': 'tts_generation',
            'reportId': report_id
        })
        
        # Use detected gender from transcription, fallback to Female
        detected_gender = transcription_data.get('speaker_gender', 'unknown')
        if detected_gender.lower() == 'male':
            tts_gender = 'Male'
        elif detected_gender.lower() == 'female':
            tts_gender = 'Female'
        else:
            tts_gender = 'Female'  # Default fallback
        
        try:
            # Generate unique filename for the enhanced audio
            timestamp = int(time.time())
            enhanced_filename = f"enhanced_speech_from_video_{report_id}_{timestamp}"
            
            success, audio_result = get_speech(
                text=enhanced_data['enhanced_text'],
                gender=tts_gender,
                filename=enhanced_filename
            )
            
            if not success:
                print(f"[ERROR] TTS generation failed: {audio_result}")
                audio_file_path = None
                tts_error = audio_result
            else:
                audio_file_path = audio_result
                tts_error = None
                
        except Exception as e:
            print(f"[ERROR] TTS generation exception: {e}")
            audio_file_path = None
            tts_error = str(e)
        
        # Step 4: Compile comprehensive results
        socketio.emit('video-audio-analysis-update', {
            'message': 'Compiling advanced analysis results...',
            'progress': 95,
            'stage': 'compilation',
            'reportId': report_id
        })
        
        # Calculate scores for presentation metrics
        original_word_count = len(transcription_data['full_text'].split()) if transcription_data['full_text'] else 0
        enhanced_word_count = len(enhanced_data['enhanced_text'].split()) if enhanced_data['enhanced_text'] else 0

        # Count filler words in original (rough estimate)
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'so', 'well']
        original_lower = transcription_data['full_text'].lower() if transcription_data['full_text'] else ""
        filler_count = sum(original_lower.count(filler) for filler in filler_words)

        # Enhanced scoring with vocal dynamics
        clarity_score = metrics_data.get('accuracy', 75) if metrics_data else 75
        pace_score = metrics_data.get('fluency', 70) if metrics_data else 70
        confidence_score = max(0, 100 - (filler_count * 5))
        engagement_score = metrics_data.get('prosody', 80) if metrics_data else 80

        # Add vocal dynamics influence to scores
        if vocal_dynamics_data:
            # Boost clarity score with vocal clarity
            vocal_clarity_boost = vocal_dynamics_data.get('vocal_health_indicators', {}).get('vocal_clarity', 0)
            clarity_score = min(100, clarity_score + (vocal_clarity_boost - 50) * 0.3)
            
            # Boost engagement with pitch variation
            pitch_variation_boost = vocal_dynamics_data.get('pitch_dynamics', {}).get('variation_score', 0)
            engagement_score = min(100, engagement_score + (pitch_variation_boost - 50) * 0.4)
            
            # Adjust pace score with rhythm analysis
            rhythm_boost = vocal_dynamics_data.get('rhythm_analysis', {}).get('rhythm_score', 0)
            pace_score = min(100, (pace_score + rhythm_boost) / 2)

        # Calculate overall score as average
        overall_score = int((clarity_score + pace_score + confidence_score + engagement_score) / 4)
        
        # Build comprehensive payload - same structure as direct audio analysis
        comprehensive_payload = {
            # Basic transcript data (matches frontend expectations)
            'transcriptSegments': transcription_data['full_text'],
            'enhancedTranscript': enhanced_data['enhanced_text'],
            'enhancedAudioUrl': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'duration': float(transcription_data.get('duration', 0)),
            'language': transcription_data.get('language', 'en'),
            'overallScore': overall_score,
            'source': 'video_extraction',  # Mark this as extracted from video
            
            # Presentation metrics (enhanced with vocal dynamics)
            'presentationMetrics': {
                'clarity_score': int(clarity_score),
                'pace_score': int(pace_score),
                'confidence_score': int(confidence_score),
                'engagement_score': int(engagement_score),
                'clarity_feedback': f"Your pronunciation accuracy is {clarity_score:.0f}%. {get_vocal_clarity_feedback(vocal_dynamics_data, clarity_score)}",
                'pace_feedback': f"Your speaking rhythm scores {pace_score:.0f}%. {get_pace_feedback(vocal_dynamics_data, pace_score)}",
                'confidence_feedback': f"Detected {filler_count} filler words. {get_confidence_feedback(vocal_dynamics_data, confidence_score)}",
                'engagement_feedback': f"Your vocal engagement scores {engagement_score:.0f}%. {get_engagement_feedback(vocal_dynamics_data, engagement_score)}"
            },
            
            # Advanced Vocal Dynamics Section
            'vocalDynamics': vocal_dynamics_data if vocal_dynamics_data else {
                'error': 'Vocal dynamics analysis failed',
                'pitch_dynamics': {'variation_score': 0, 'dynamic_range': 'Analysis Failed'},
                'volume_dynamics': {'energy_score': 0, 'dynamic_presence': 'Unknown'},
                'rhythm_analysis': {'rhythm_score': 0, 'pace_category': 'Unknown'},
                'pause_analysis': {'strategic_score': 0, 'pause_effectiveness': 'Unknown'},
                'overall_dynamics_score': 0,
                'presentation_readiness': {'readiness_level': 'Analysis Failed'}
            },
            
            # Enhancement data (matches frontend structure)
            'enhancement': {
                'key_changes': enhanced_data.get('improvements_made', []),
                'speaking_tips': enhanced_data.get('presentation_tips', []) + get_vocal_dynamics_tips(vocal_dynamics_data),
                'summary': f"Your speech has been enhanced by removing filler words and improving flow. Word count changed from {original_word_count} to {enhanced_word_count} words. (Source: Video extraction)"
            },
            
            # Speech analysis data (enhanced with vocal metrics)
            'speechAnalysis': {
                'word_count': original_word_count,
                'speaking_rate': int((original_word_count / transcription_data.get('duration', 1)) * 60) if transcription_data.get('duration', 0) > 0 else 0,
                'pace_feedback': f"Speaking rate: {int((original_word_count / transcription_data.get('duration', 1)) * 60)} words per minute. {get_detailed_pace_feedback(vocal_dynamics_data)}",
                'filler_feedback': f"Detected {filler_count} filler words. Focus on pausing instead of using 'um', 'uh', 'like', etc.",
                'vocal_summary': get_vocal_summary(vocal_dynamics_data)
            },
            
            # Technical metadata (for debugging/additional info)
            'technicalData': {
                'transcription_metadata': {
                    'language': transcription_data.get('language', 'unknown'),
                    'duration': float(transcription_data.get('duration', 0)),
                    'speaker_gender': transcription_data.get('speaker_gender', 'unknown'),
                    'pitch_hz': float(transcription_data.get('pitch_hz', 0)),
                    'segments': transcription_data.get('segments', [])
                },
                'speech_metrics': metrics_data if metrics_data else {
                    'error': 'Metrics analysis failed',
                    'pronunciation': 0,
                    'accuracy': 0,
                    'fluency': 0,
                    'prosody': 0
                },
                'vocal_dynamics_raw': vocal_dynamics_data,
                'generated_audio': {
                    'file_path': audio_file_path,
                    'voice_gender': tts_gender,
                    'generation_success': audio_file_path is not None,
                    'error': tts_error
                },
                'extraction_info': {
                    'source': 'video',
                    'report_id': report_id,
                    'extraction_method': 'ffmpeg'
                },
                'component_status': {
                    'metrics': {
                        'success': metrics_data is not None,
                        'error': metrics_error
                    },
                    'vocal_dynamics': {
                        'success': vocal_dynamics_data is not None,
                        'error': vocal_dynamics_error
                    },
                    'transcription': {
                        'success': transcription_data is not None,
                        'error': transcription_error
                    },
                    'enhancement': {
                        'success': enhancement_error is None,
                        'error': enhancement_error
                    },
                    'tts_generation': {
                        'success': audio_file_path is not None,
                        'error': tts_error
                    }
                }
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        comprehensive_payload = convert_numpy_types(comprehensive_payload)
        
        # Final success emission with enhanced summary
        summary_data = {
            'components_completed': sum([
                metrics_data is not None,
                vocal_dynamics_data is not None,
                transcription_data is not None,
                enhancement_error is None,
                audio_file_path is not None
            ]),
            'total_components': 5,
            'audio_file_location': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'analysis_success': True,
            'overall_score': comprehensive_payload['overallScore'],
            'source': 'video_extraction',
            'report_id': report_id,
            'key_metrics': {
                'word_count': original_word_count,
                'duration': float(transcription_data.get('duration', 0)),
                'filler_count': filler_count,
                'enhancement_available': audio_file_path is not None,
                'vocal_dynamics_score': vocal_dynamics_data.get('overall_dynamics_score', 0) if vocal_dynamics_data else 0,
                'presentation_readiness': vocal_dynamics_data.get('presentation_readiness', {}).get('readiness_level', 'Unknown') if vocal_dynamics_data else 'Unknown'
            }
        }
        
        # Convert summary data as well
        summary_data = convert_numpy_types(summary_data)
        
        socketio.emit('video-audio-analysis-complete', {
            'message': 'Video-to-audio analysis complete!',
            'progress': 100,
            'stage': 'complete',
            'reportId': report_id,
            'data': comprehensive_payload,
            'summary': summary_data
        })
        
        print(f"[INFO] Video-to-audio analysis completed successfully for report {report_id}")
        
    except Exception as e:
        print(f"[ERROR] Video-to-audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_data = {
            'error': f'Video-to-audio processing failed: {str(e)}',
            'stage': 'processing',
            'reportId': report_id
        }
        error_data = convert_numpy_types(error_data)
        socketio.emit('video-audio-analysis-error', error_data)
    
    finally:
        # Cleanup temporary file
        try:
            time.sleep(1)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[INFO] Cleaned up temporary audio file: {temp_path}")
        except PermissionError:
            time.sleep(2)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[INFO] Cleaned up temporary audio file (delayed): {temp_path}")
            except Exception as e:
                print(f"[WARNING] Could not cleanup temp audio file: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp audio file: {e}")

# Add a cleanup function to remove old video file references
def cleanup_old_video_files():
    """Clean up video file tracker entries older than 2 hours"""
    current_time = time.time()
    cutoff_time = current_time - 7200  # 2 hours
    
    to_remove = []
    for report_id, file_path in video_file_tracker.items():
        try:
            # Extract timestamp from report_id
            timestamp_str = report_id.split('_')[1]
            timestamp = int(timestamp_str)
            
            if timestamp < cutoff_time:
                to_remove.append(report_id)
        except (IndexError, ValueError):
            # If we can't parse the timestamp, remove it
            to_remove.append(report_id)
    
    for report_id in to_remove:
        video_file_tracker.pop(report_id, None)
        print(f"[INFO] Cleaned up old video file reference: {report_id}")

#  helper endpoint for debugging
@app.route('/api/debug/video-tracker')
def debug_video_tracker():
    """Debug endpoint to see tracked video files"""
    return jsonify({
        'tracked_files': len(video_file_tracker),
        'files': {k: v for k, v in video_file_tracker.items()}
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=True)
