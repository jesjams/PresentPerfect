"""
Audio utilities:
1. Robust audio loading that prefers PySoundFile (libsndfile) and falls back to audioread.
2. Feature extraction for basic gender detection.
3. Simple rule-based gender classifier (pitch-based).
4. Whisper transcription wrapper that also returns speaker gender and mean pitch.

Dependencies
------------
conda install -c conda-forge libsndfile soundfile
pip install torch whisper librosa numpy
"""

import warnings
import numpy as np
import torch
import librosa
import soundfile as sf
import whisper
import subprocess, tempfile, os
from pathlib import Path
from typing import Tuple
# ----------------------------- configuration -----------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET_SR = 22_050   # common sample rate for feature extraction
FFMPEG_EXE = "ffmpeg"
# -------------------------------------------------------------------------

def _convert_to_wav(source: str, sr: int = TARGET_SR) -> str:
    """
    Convert `source` to mono WAV at `sr` Hz via FFmpeg, return new path.
    If source is already WAV/FLAC/OGG (valid for libsndfile) it is returned
    unchanged.
    """
    ext = Path(source).suffix.lower()
    if ext not in {".m4a", ".mp4", ".aac", ".mp3", ".wma"}:
        return source  # nothing to do

    tmp_path = Path(tempfile.gettempdir()) / (Path(source).stem + "_tmp.wav")
    cmd = [
        FFMPEG_EXE, "-y", "-i", source,  # input
        "-ac", "1",                      # mono
        "-ar", str(sr),                  # sample rate
        tmp_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg not found – install it or place on PATH") from e
    return str(tmp_path)
# -------------------------------------------------------------------------
# Robust audio loader
# -------------------------------------------------------------------------
def robust_load(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Returns mono float32 PCM at `sr` Hz.
    Uses PySoundFile (libsndfile) where possible and falls back to
    librosa+audioread when libsndfile cannot handle the codec.
    """
    path = _convert_to_wav(path, sr)  # ensure libsndfile-friendly

    try:
        y, native_sr = sf.read(path, dtype="float32", always_2d=False)
        if native_sr != sr:
            y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    except Exception as err:
        # libs/codec not supported – fall back to audioread
        warnings.warn(f"SoundFile read failed ({err}); using audioread",
                      RuntimeWarning)
        y, _ = librosa.load(path, sr=sr, mono=True, res_type="kaiser_fast")
    return y, sr

# -------------------------------------------------------------------------
# Feature extraction for gender detection
# -------------------------------------------------------------------------
def extract_gender_features(audio_path: str):
    """
    Extracts a vector of pitch, spectral, MFCC and ZCR statistics.
    Returns a list of floats or None on error.
    """
    try:
        y, sr = robust_load(audio_path, sr=TARGET_SR)

        feats = {}

        # --- fundamental frequency via piptrack ---
        pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
        pitches = pitches[pitches > 0]
        if pitches.size:
            feats.update(
                pitch_mean=np.mean(pitches),
                pitch_std=np.std(pitches),
                pitch_min=np.min(pitches),
                pitch_max=np.max(pitches),
            )
        else:
            feats.update(
                pitch_mean=0,
                pitch_std=0,
                pitch_min=0,
                pitch_max=0,
            )

        # --- spectral centroid ---
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats.update(
            spectral_centroid_mean=np.mean(cent),
            spectral_centroid_std=np.std(cent),
        )

        # --- MFCCs (13 coefficients) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            feats[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
            feats[f"mfcc_{i}_std"] = np.std(mfcc[i])

        # --- zero-crossing rate ---
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats.update(
            zcr_mean=np.mean(zcr),
            zcr_std=np.std(zcr),
        )
        return list(feats.values())
    except Exception as err:
        print(f"[ERROR] extract_gender_features: {err}")
        return None


# -------------------------------------------------------------------------
# Simple rule-based gender classifier
# -------------------------------------------------------------------------
def simple_gender_classifier(audio_path: str):
    """
    Returns (gender, mean_pitch_hz)
    gender ∈ {'male', 'female', 'unknown'}
    """
    try:
        y, sr = robust_load(audio_path, sr=TARGET_SR)

        # Fundamental frequency estimation (try YIN first)
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            valid = f0[(f0 > 50) & (f0 < 400) & ~np.isnan(f0)]
        except Exception:
            pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
            valid = pitches[(pitches > 50) & (pitches < 400)]

        if valid.size == 0:
            return "unknown", 0.0

        mean_pitch = float(np.mean(valid))

        # Basic thresholding
        if mean_pitch < 165:
            return "male", mean_pitch
        if mean_pitch <= 265:
            return "female", mean_pitch
        return "unknown", mean_pitch
    except Exception as err:
        print(f"[ERROR] simple_gender_classifier: {err}")
        return "unknown", 0.0


# -------------------------------------------------------------------------
# Whisper transcription + gender metadata
# -------------------------------------------------------------------------
def whisper_transcriber(path: str):
    """
    Transcribes `path` with OpenAI Whisper and augments result with
    gender + mean pitch.
    """
    try:
        model = whisper.load_model("turbo", device=DEVICE)

        result = model.transcribe(path, fp16=False, verbose=False)

        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        duration = max((seg.get("end", 0) for seg in segments), default=0)

        gender, pitch = simple_gender_classifier(path)

        return {
            "segments": segments,
            "full_text": full_text,
            "language": language,
            "duration": duration,
            "speaker_gender": gender,
            "pitch_hz": pitch,
        }
    except Exception as err:
        print(f"[ERROR] whisper_transcriber: {err}")
        return None


# -------------------------------------------------------------------------
# Optional: silence the librosa futurewarning noise during dev
# -------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="librosa",
)
