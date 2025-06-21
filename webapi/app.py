# ──────────────────────────────  Imports & Monkey-patch  ──────────────────────────────
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, send_from_directory, jsonify
from pathlib import Path
from flask_socketio import SocketIO
from flask_cors import CORS

import os, time, tempfile, random, queue, threading, math
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


    # Build system + user messages
    system = (
    "You are an experienced presentation coach. Your reply **must be a single JSON object** that follows the exact structure of the PresentationFeedback schema shown below—no extra keys, text, or formatting.\n\n"
    "PresentationFeedback schema:\n"
    "{\n"
    "  'speechImprovements\": string,\n"
    "  'speechScore\":       integer,   // 1-100\n"
    "  'emotionScore\":      integer,   // 1-100\n"
    "  'emotionText\":       string,\n"
    "  'gazeScore\":         integer,   // 1-100\n"
    "  'gazeText\":          string,\n"
    "  'movementScore\":     integer,   // 1-100\n"
    "  'movementText\":      string,\n"
    "  'shoulderScore\":     integer,   // 1-100\n"
    "  'shoulderText\":      string,\n"
    "  'handsScore\":        integer,   // 1-100\n"
    "  'gestureText\":       string,\n"
    "  'overallScore\":      integer,   // 1-100  (average of the five sub-scores)\n"
    "  'overallSummary':    string\n"
    "}\n\n"
    "INSTRUCTIONS\n"
    "1. Read the transcript to understand the presentation’s content and intent.\n"
    "2. Use the analytics timeline to assess the presenter’s delivery.\n"
    "3. Fill every field of the PresentationFeedback JSON:\n"
    "   • *speechScore* – quality of wording, structure, clarity (1-100). A good presentation script should be clear, structured, and support the presentation intent.\n"
    "   • *emotionScore* – how well facial emotion matches the script (1-100). A good presenter should use their emotion to support their presentation intent.\n"
    "   • *gazeScore* – audience engagement through eye contact (1-100). A good presenter engages their audience by not focusing their gaze on only one spot during their presentation.\n"
    "   • *movementScore* – purposeful use of stage space (1-100). A good presenter uses their stage effectively; they should not move too little or too much.\n"
    "   • *shoulderScore* – confident posture (1-100). A good presenter should be confident and appear reliable.\n"
    "   • *handsScore* – effective hand gestures (1-100). A good presenter uses their gestures effectively to deliver their presentation.\n"
    "   • Provide detailed text recommendations for each area.\n"
    "4. Calculate *overallScore* as the average of speechScore, emotionScore, gazeScore, movementScore, shoulderScore, and handsScore.\n"
    "5. Summarise the key action items in *overallSummary*.\n\n"
    "**Return only the JSON object that conforms to the schema.**"
)
    user = f"""
Below are (1) second‑by‑second analytics extracted from the video and (2) the full speech transcript.
 
-----------------------------------------------
ANALYTICS  (one entry per second)
  • Emotion compiled by transcript speaking time : {dom_emotion}      // dominant facial emotion
  • Gaze_sec    : {dom_gaze}         // gaze region: centre, up, down, left, right, upleft, upright, downleft, downright
  • Move_avg_sec: {move_avg}         // X‑axis position 0‑10 (0 = far left, 10 = far right)
  • Shoulder_sec: {dom_shoulder}     // posture flag: slouch / upright
  • Hands_sec   : {dom_hands}        // gesture flag: gesturing / static
-----------------------------------------------
 
TRANSCRIPT
{segments}

Respond with ONLY a JSON object matching the PresentationFeedback model.
"""

    # Request and parse
    completion = client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL"),                 
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        response_format=PresentationFeedback
    )

    # Extract the parsed model
    feedback: PresentationFeedback = completion.choices[0].message.parsed
    return feedback

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
    reset_state()
    video = request.files['video']
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
    """Process audio with metrics, transcription, enhancement, and TTS generation"""
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
        
        # Step 1: Run metrics and transcription in parallel
        socketio.emit('audio-analysis-update', {
            'message': 'Analyzing speech metrics and transcribing audio...',
            'progress': 20,
            'stage': 'parallel_analysis'
        })
        
        executor = ThreadPoolExecutor(max_workers=2)
        
        # Create wrapper functions with better error handling
        def safe_get_metrics(path):
            try:
                print(f"[INFO] Starting metrics analysis for: {path}")
                # Check if file exists and has content
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
                            # Clean up converted file
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
                # Ensure file is accessible before transcription
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
        
        # Submit both tasks with the safe wrappers
        metrics_future = executor.submit(safe_get_metrics, temp_path)
        transcription_future = executor.submit(safe_whisper_transcriber, temp_path)
        
        # Wait for metrics
        socketio.emit('audio-analysis-update', {
            'message': 'Processing speech quality metrics...',
            'progress': 35,
            'stage': 'metrics'
        })
        
        metrics_error = None
        try:
            metrics_data = metrics_future.result(timeout=60)  # 1 minute timeout
        except Exception as e:
            print(f"[ERROR] Metrics analysis failed: {e}")
            metrics_data = None
            metrics_error = str(e)
        
        # Wait for transcription
        socketio.emit('audio-analysis-update', {
            'message': 'Completing transcription and gender detection...',
            'progress': 50,
            'stage': 'transcription'
        })
        
        transcription_error = None
        try:
            transcription_data = transcription_future.result(timeout=120)  # 2 minute timeout
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
                    'transcription_success': False,
                    'transcription_error': transcription_error
                }
            }
            # Convert any numpy types in error data
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
            'message': 'Compiling analysis results...',
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

        # Calculate scores for presentation metrics (now that filler_count is available)
        clarity_score = metrics_data.get('accuracy', 75) if metrics_data else 75
        pace_score = metrics_data.get('fluency', 70) if metrics_data else 70
        confidence_score = max(0, 100 - (filler_count * 5))  # Reduce score based on filler words
        engagement_score = metrics_data.get('prosody', 80) if metrics_data else 80

        # Calculate overall score as average
        overall_score = int((clarity_score + pace_score + confidence_score + engagement_score) / 4)
        
        # Build comprehensive payload to match frontend expectations
        comprehensive_payload = {
            # Basic transcript data (matches frontend expectations)
            'transcriptSegments': transcription_data['full_text'],  # Frontend expects this key
            'enhancedTranscript': enhanced_data['enhanced_text'],
            'enhancedAudioUrl': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'duration': float(transcription_data.get('duration', 0)),
            'language': transcription_data.get('language', 'en'),
            'overallScore': overall_score,  # Use calculated overall score
            
            # Presentation metrics (matches frontend structure)
            'presentationMetrics': {
                'clarity_score': clarity_score,
                'pace_score': pace_score,
                'confidence_score': confidence_score,
                'engagement_score': engagement_score,
                'clarity_feedback': f"Your pronunciation accuracy is {clarity_score}%. Focus on clear articulation of words." if metrics_data else "Work on speaking clearly and articulating each word properly.",
                'pace_feedback': f"Your fluency score is {pace_score}%. Try to maintain a steady speaking rhythm." if metrics_data else "Focus on maintaining an appropriate speaking pace - not too fast, not too slow.",
                'confidence_feedback': f"Detected {filler_count} filler words. Reduce 'um', 'uh', and similar words to sound more confident.",
                'engagement_feedback': f"Your prosody score is {engagement_score}%. Use more vocal variety to engage listeners." if metrics_data else "Add more energy and enthusiasm to capture audience attention."
            },
            
            # Enhancement data (matches frontend structure)
            'enhancement': {
                'key_changes': enhanced_data.get('improvements_made', []),
                'speaking_tips': enhanced_data.get('presentation_tips', []),
                'summary': f"Your speech has been enhanced by removing filler words and improving flow. Word count changed from {original_word_count} to {enhanced_word_count} words."
            },
            
            # Speech analysis data (matches frontend structure)
            'speechAnalysis': {
                'word_count': original_word_count,
                'speaking_rate': int((original_word_count / transcription_data.get('duration', 1)) * 60) if transcription_data.get('duration', 0) > 0 else 0,
                'pace_feedback': f"Speaking rate: {int((original_word_count / transcription_data.get('duration', 1)) * 60)} words per minute. Ideal range is 140-160 WPM." if transcription_data.get('duration', 0) > 0 else "Unable to calculate speaking rate.",
                'filler_feedback': f"Detected {filler_count} filler words. Focus on pausing instead of using 'um', 'uh', 'like', etc."
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
                    'has_enhancement': enhanced_data is not None,
                    'has_generated_audio': audio_file_path is not None,
                    'overall_success': all([
                        transcription_data is not None,  # This is essential
                        # Others are optional but tracked
                    ])
                }
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        comprehensive_payload = convert_numpy_types(comprehensive_payload)
        
        # Final success emission with frontend-compatible summary
        summary_data = {
            'components_completed': sum([
                metrics_data is not None,
                transcription_data is not None,
                enhancement_error is None,
                audio_file_path is not None
            ]),
            'total_components': 4,
            'audio_file_location': f'/static/generated_audio/{audio_file_path}' if audio_file_path else None,
            'analysis_success': True,
            'overall_score': comprehensive_payload['overallScore'],
            'key_metrics': {
                'word_count': original_word_count,
                'duration': float(transcription_data.get('duration', 0)),
                'filler_count': filler_count,
                'enhancement_available': audio_file_path is not None
            }
        }
        
        # Convert summary data as well
        summary_data = convert_numpy_types(summary_data)
        
        socketio.emit('audio-analysis-complete', {
            'message': 'Comprehensive audio analysis complete!',
            'progress': 100,
            'stage': 'complete',
            'data': comprehensive_payload,
            'summary': summary_data
        })
        
        print("[INFO] Comprehensive audio analysis completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Comprehensive audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_data = {
            'error': f'Processing failed: {str(e)}',
            'stage': 'processing'
        }
        # Convert any numpy types in error data
        error_data = convert_numpy_types(error_data)
        socketio.emit('audio-analysis-error', error_data)
    
    finally:
        # Cleanup temporary file
        try:
            # Wait a bit to ensure all processes are done with the file
            time.sleep(1)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[INFO] Cleaned up temporary file: {temp_path}")
        except PermissionError:
            # File might still be in use, try again after a delay
            time.sleep(2)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[INFO] Cleaned up temporary file (delayed): {temp_path}")
            except Exception as e:
                print(f"[WARNING] Could not cleanup temp file: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to cleanup temp file: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=True)
