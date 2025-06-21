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

#Models & Consts  
MODEL_PATH   = os.getenv("MODEL_PATH") or "yolov8n.pt"
emotion_model = YOLO(MODEL_PATH)

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

STRAIGHT_THRESHOLD_DEG = 7.0  
GESTURE_RATIO          = 0.22

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

def get_direction(yaw, pitch):
    if yaw >  YAW_THR:
        h = "right"
    elif yaw < -YAW_THR:
        h = "left"
    else:
        h = ""
    if 180 >= pitch >= 160 or -180 <= pitch <= -160:
        v = ""
    else:
        v = "down" if pitch < 0 else "up"
    if not h and not v:
        return "straight"
    if not v:
        return h
    if not h:
        return v
    return f"{v}-{h}"

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
        model="gpt-4.1-nano-2025-04-14",                 
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

        #HAND STATE  (gesturing vs at-side)
        l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        hip_mid_x = (l_hp.x + r_hp.x) / 2.0
        hip_mid_y = (l_hp.y + r_hp.y) / 2.0

        dist_l = math.hypot(l_wr.x - hip_mid_x, l_wr.y - hip_mid_y)
        dist_r = math.hypot(r_wr.x - hip_mid_x, r_wr.y - hip_mid_y)
        is_gesturing = max(dist_l, dist_r) > GESTURE_RATIO
        label_hands  = "Gesturing" if is_gesturing else "Idle Hands"
        gesture_per_second[sec].append(label_hands)

def gaze_batch(batch_rgbs, batch_secs, CAM_MAT, DIST, W, H):
    for img_rgb, sec in zip(batch_rgbs, batch_secs):
        # process frame for face landmarks
        res = face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            # no face detected: assign center gaze
            gaze_per_second[sec].append(get_direction(0.0, 0.0))
            continue

        lm = res.multi_face_landmarks[0]
        pts2d = np.array([
            (lm.landmark[LANDMARK_IDS["nose_tip"]].x  * W,
             lm.landmark[LANDMARK_IDS["nose_tip"]].y  * H),
            (lm.landmark[LANDMARK_IDS["chin"]].x       * W,
             lm.landmark[LANDMARK_IDS["chin"]].y       * H),
            (lm.landmark[LANDMARK_IDS["left_eye_outer"]].x  * W,
             lm.landmark[LANDMARK_IDS["left_eye_outer"]].y  * H),
            (lm.landmark[LANDMARK_IDS["right_eye_outer"]].x * W,
             lm.landmark[LANDMARK_IDS["right_eye_outer"]].y * H),
            (lm.landmark[LANDMARK_IDS["mouth_left"]].x  * W,
             lm.landmark[LANDMARK_IDS["mouth_left"]].y  * H),
            (lm.landmark[LANDMARK_IDS["mouth_right"]].x * W,
             lm.landmark[LANDMARK_IDS["mouth_right"]].y * H)
        ], dtype=np.float64)

        # estimate head pose
        ok, rvec, _ = cv2.solvePnP(
            G_MODEL_POINTS, pts2d, CAM_MAT, DIST,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            # PnP failed: assign center gaze
            gaze_per_second[sec].append(get_direction(0.0, 0.0))
            continue

        # convert rotation vector to matrix and extract angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw, pitch = angles[1], angles[0]

        # append the computed gaze direction
        gaze_per_second[sec].append(get_direction(yaw, pitch))

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
            pool.submit(movement_batch, rgbs, secs)
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


@app.route('/api/analyze-audio', methods=['POST'])
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
            
        actual_size = os.path.getsize(temp_path)
        print(f"[INFO] Audio saved: {actual_size / 1024:.2f} KB")
        
        if actual_size == 0:
            return {'error': 'Audio file is empty'}, 400

        # Check if enhanced audio processor is available
        if enhanced_audio_processor is None:
            return {'error': 'Audio processor not available. Check server configuration.'}, 500

        # Use enhanced audio processing
        socketio.start_background_task(process_audio_for_presentation, temp_path, enhanced_audio_processor, socketio)
        return {'status': 'audio processing started'}

    except Exception as e:
        print(f"[ERROR] analyze_audio endpoint failed: {e}")
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=True)
