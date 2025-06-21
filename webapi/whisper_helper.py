import whisper
import librosa
import numpy as np
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_gender_features(audio_path):
    """Extract audio features for gender detection"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract features
        features = {}
        
        # Fundamental frequency (pitch) statistics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            features['pitch_mean'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)
            features['pitch_min'] = np.min(pitches)
            features['pitch_max'] = np.max(pitches)
        else:
            features['pitch_mean'] = features['pitch_std'] = 0
            features['pitch_min'] = features['pitch_max'] = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return list(features.values())
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def simple_gender_classifier(audio_path):
    """Simple rule-based gender detection using pitch"""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        
        if len(pitches) > 0:
            mean_pitch = np.mean(pitches)
            # Simple threshold-based classification
            # Typical male: 85-180 Hz, Female: 165-265 Hz
            if mean_pitch < 165:
                return 'male', mean_pitch
            else:
                return 'female', mean_pitch
        else:
            return 'unknown', 0
    except Exception as e:
        print(f"Error in gender classification: {e}")
        return 'unknown', 0

def whisper_transcriber(path):
    try:
        whisper_model = whisper.load_model("turbo", device=DEVICE)        
        result = whisper_model.transcribe(path, fp16=False, verbose=False)
        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        duration = max(seg.get('end', 0) for seg in segments) if segments else 0
        gender, pitch = simple_gender_classifier(path)
        data = {
            'segments': segments,
            'full_text': full_text,
            'language': language,
            'duration': duration,
            'speaker_gender': gender,
            'pitch_hz': pitch
        }
        return data
    except Exception as e:
        print(f"[ERROR] Failed to transcribe audio: {e}")
        return None
        