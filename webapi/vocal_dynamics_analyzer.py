# import librosa
# import numpy as np
# import scipy.stats
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from typing import Dict, List, Tuple, Any

# def analyze_vocal_dynamics(audio_path: str) -> Dict[str, Any]:
#     """
#     Advanced vocal dynamics analysis for presentation coaching
#     Returns sophisticated metrics about pitch variation, volume dynamics, rhythm, and pauses
#     """
#     try:
#         # Load audio with higher sample rate for better analysis
#         y, sr = librosa.load(audio_path, sr=22050)
#         duration = len(y) / sr
        
#         # 1. PITCH VARIATION ANALYSIS
#         pitch_metrics = analyze_pitch_variation(y, sr)
        
#         # 2. VOLUME DYNAMICS ANALYSIS  
#         volume_metrics = analyze_volume_dynamics(y, sr)
        
#         # 3. SPEAKING RHYTHM ANALYSIS
#         rhythm_metrics = analyze_speaking_rhythm(y, sr)
        
#         # 4. STRATEGIC PAUSE ANALYSIS
#         pause_metrics = analyze_pauses(y, sr)
        
#         # 5. OVERALL VOCAL DYNAMICS SCORE
#         overall_score = calculate_vocal_dynamics_score(
#             pitch_metrics, volume_metrics, rhythm_metrics, pause_metrics
#         )
        
#         return {
#             'duration': duration,
#             'pitch_dynamics': pitch_metrics,
#             'volume_dynamics': volume_metrics,
#             'rhythm_analysis': rhythm_metrics,
#             'pause_analysis': pause_metrics,
#             'overall_dynamics_score': overall_score,
#             'vocal_health_indicators': analyze_vocal_health(y, sr),
#             'presentation_readiness': calculate_presentation_readiness(
#                 pitch_metrics, volume_metrics, rhythm_metrics, pause_metrics
#             )
#         }
        
#     except Exception as e:
#         print(f"[ERROR] Vocal dynamics analysis failed: {e}")
#         return None

# def analyze_pitch_variation(y: np.ndarray, sr: int) -> Dict[str, Any]:
#     """Analyze pitch variation and monotone detection"""
    
#     # Extract fundamental frequency using piptrack (more robust than yin for this)
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
    
#     # Get pitch values over time
#     pitch_values = []
#     times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
    
#     for t in range(pitches.shape[1]):
#         index = magnitudes[:, t].argmax()
#         pitch = pitches[index, t]
#         if pitch > 0:  # Valid pitch
#             pitch_values.append(pitch)
    
#     if len(pitch_values) < 10:  # Not enough data
#         return {
#             'variation_score': 0,
#             'monotone_risk': 100,
#             'pitch_range_hz': 0,
#             'pitch_stability': 0,
#             'dynamic_range': 'Very Limited'
#         }
    
#     pitch_array = np.array(pitch_values)
    
#     # Calculate pitch variation metrics
#     pitch_std = np.std(pitch_array)
#     pitch_range = np.max(pitch_array) - np.min(pitch_array)
#     pitch_mean = np.mean(pitch_array)
    
#     # Pitch variation score (0-100, higher = more dynamic)
#     # Good speakers have 15-30% pitch variation
#     variation_coefficient = pitch_std / pitch_mean if pitch_mean > 0 else 0
#     variation_score = min(100, variation_coefficient * 300)  # Scale to 0-100
    
#     # Monotone risk (0-100, higher = more monotone)
#     monotone_risk = max(0, 100 - variation_score)
    
#     # Pitch stability (consistency without being monotone)
#     pitch_stability = 100 - (abs(variation_coefficient - 0.2) * 250)  # Optimal around 20%
#     pitch_stability = max(0, min(100, pitch_stability))
    
#     # Dynamic range classification
#     if variation_score > 80:
#         dynamic_range = "Highly Dynamic"
#     elif variation_score > 60:
#         dynamic_range = "Well Varied"
#     elif variation_score > 40:
#         dynamic_range = "Moderately Varied"
#     elif variation_score > 20:
#         dynamic_range = "Somewhat Monotone"
#     else:
#         dynamic_range = "Very Monotone"
    
#     return {
#         'variation_score': int(variation_score),
#         'monotone_risk': int(monotone_risk),
#         'pitch_range_hz': round(pitch_range, 1),
#         'pitch_stability': int(pitch_stability),
#         'dynamic_range': dynamic_range,
#         'mean_pitch': round(pitch_mean, 1),
#         'pitch_coefficient_variation': round(variation_coefficient * 100, 1)
#     }

# def analyze_volume_dynamics(y: np.ndarray, sr: int) -> Dict[str, Any]:
#     """Analyze volume/energy dynamics throughout speech"""
    
#     # Calculate RMS energy in windows
#     frame_length = 2048
#     hop_length = 512
#     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
#     # Convert to dB for more meaningful analysis
#     rms_db = 20 * np.log10(rms + 1e-6)  # Add small value to avoid log(0)
    
#     # Volume metrics
#     volume_range = np.max(rms_db) - np.min(rms_db)
#     volume_std = np.std(rms_db)
#     mean_volume = np.mean(rms_db)
    
#     # Energy variation score
#     energy_score = min(100, volume_std * 3)  # Scale to 0-100
    
#     # Volume consistency (not too varied, not too flat)
#     ideal_range = 12  # dB - good speakers have ~12dB dynamic range
#     consistency_score = 100 - abs(volume_range - ideal_range) * 4
#     consistency_score = max(0, min(100, consistency_score))
    
#     # Detect volume patterns
#     volume_trend = analyze_volume_trend(rms_db)
    
#     # Energy sustainability (how well energy is maintained)
#     sustainability = calculate_energy_sustainability(rms_db)
    
#     return {
#         'energy_score': int(energy_score),
#         'volume_range_db': round(volume_range, 1),
#         'consistency_score': int(consistency_score),
#         'mean_volume_db': round(mean_volume, 1),
#         'energy_sustainability': int(sustainability),
#         'volume_trend': volume_trend,
#         'dynamic_presence': 'Strong' if energy_score > 70 else 'Moderate' if energy_score > 40 else 'Weak'
#     }

# def analyze_speaking_rhythm(y: np.ndarray, sr: int) -> Dict[str, Any]:
#     """Analyze speaking rhythm and tempo variations"""
    
#     # Detect speech segments using spectral features
#     hop_length = 512
    
#     # Use spectral centroid and rolloff to identify speech vs silence
#     spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
#     # Combine features to detect speech activity
#     speech_activity = (spectral_centroids > np.percentile(spectral_centroids, 25)) & \
#                      (spectral_rolloff > np.percentile(spectral_rolloff, 25))
    
#     # Calculate tempo and rhythm metrics
#     rhythm_score = calculate_rhythm_score(speech_activity, sr, hop_length)
    
#     # Pace analysis (words per minute estimation from audio features)
#     estimated_pace = estimate_speaking_pace(speech_activity, sr, hop_length)
    
#     # Rush detection (identifying sections that are too fast)
#     rush_sections = detect_rushed_sections(speech_activity, sr, hop_length)
    
#     return {
#         'rhythm_score': int(rhythm_score),
#         'estimated_pace_wpm': int(estimated_pace),
#         'pace_category': categorize_pace(estimated_pace),
#         'rush_risk': int(len(rush_sections) * 10),  # Higher if more rushed sections
#         'rhythm_consistency': calculate_rhythm_consistency(speech_activity),
#         'speech_timing': 'Excellent' if rhythm_score > 80 else 'Good' if rhythm_score > 60 else 'Needs Work'
#     }

# def analyze_pauses(y: np.ndarray, sr: int) -> Dict[str, Any]:
#     """Analyze strategic pause placement and effectiveness"""
    
#     # Detect silence/pause regions
#     frame_length = 2048
#     hop_length = 512
    
#     # Use RMS energy to detect pauses
#     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
#     # Threshold for silence (adaptive based on audio)
#     silence_threshold = np.percentile(rms, 20)  # Bottom 20% considered silence
    
#     # Find pause regions
#     is_silence = rms < silence_threshold
#     pause_regions = find_pause_regions(is_silence, sr, hop_length)
    
#     # Analyze pause characteristics
#     pause_metrics = calculate_pause_metrics(pause_regions, len(y) / sr)
    
#     # Strategic pause score (based on placement and duration)
#     strategic_score = calculate_strategic_pause_score(pause_regions)
    
#     return {
#         'total_pauses': len(pause_regions),
#         'average_pause_duration': round(pause_metrics['avg_duration'], 2),
#         'pause_frequency': round(pause_metrics['frequency'], 2),  # pauses per minute
#         'strategic_score': int(strategic_score),
#         'pause_effectiveness': 'Excellent' if strategic_score > 80 else 'Good' if strategic_score > 60 else 'Needs Improvement',
#         'longest_pause': round(pause_metrics['max_duration'], 2),
#         'pause_distribution': pause_metrics['distribution']
#     }

# def calculate_vocal_dynamics_score(pitch_metrics: Dict, volume_metrics: Dict, 
#                                  rhythm_metrics: Dict, pause_metrics: Dict) -> int:
#     """Calculate overall vocal dynamics score"""
    
#     # Weight different components
#     weights = {
#         'pitch': 0.3,
#         'volume': 0.25, 
#         'rhythm': 0.25,
#         'pauses': 0.2
#     }
    
#     pitch_score = pitch_metrics['variation_score']
#     volume_score = volume_metrics['consistency_score']
#     rhythm_score = rhythm_metrics['rhythm_score']
#     pause_score = pause_metrics['strategic_score']
    
#     overall = (pitch_score * weights['pitch'] + 
#               volume_score * weights['volume'] + 
#               rhythm_score * weights['rhythm'] + 
#               pause_score * weights['pauses'])
    
#     return int(overall)

# def analyze_vocal_health(y: np.ndarray, sr: int) -> Dict[str, Any]:
#     """Analyze vocal health indicators"""
    
#     # Spectral features that indicate vocal strain or health
#     spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
#     # Voice quality indicators
#     breathiness = np.mean(zero_crossing_rate) * 1000  # Higher = more breathy
#     clarity = np.mean(spectral_centroids) / 1000  # Voice clarity indicator
    
#     return {
#         'vocal_clarity': min(100, int(clarity * 20)),
#         'breathiness_level': min(100, int(breathiness * 50)),
#         'vocal_strain_risk': 'Low' if breathiness < 0.1 else 'Moderate' if breathiness < 0.2 else 'High'
#     }

# def calculate_presentation_readiness(pitch_metrics: Dict, volume_metrics: Dict, 
#                                    rhythm_metrics: Dict, pause_metrics: Dict) -> Dict[str, Any]:
#     """Calculate overall presentation readiness metrics"""
    
#     # Professional presentation benchmarks
#     benchmarks = {
#         'pitch_variation': 60,  # Good speakers have 60+ variation score
#         'volume_consistency': 70,  # Should have good volume control
#         'rhythm_score': 65,  # Steady rhythm
#         'pause_effectiveness': 60  # Strategic pauses
#     }
    
#     readiness_scores = {
#         'pitch_readiness': min(100, (pitch_metrics['variation_score'] / benchmarks['pitch_variation']) * 100),
#         'volume_readiness': min(100, (volume_metrics['consistency_score'] / benchmarks['volume_consistency']) * 100),
#         'rhythm_readiness': min(100, (rhythm_metrics['rhythm_score'] / benchmarks['rhythm_score']) * 100),
#         'pause_readiness': min(100, (pause_metrics['strategic_score'] / benchmarks['pause_effectiveness']) * 100)
#     }
    
#     overall_readiness = sum(readiness_scores.values()) / len(readiness_scores)
    
#     return {
#         **readiness_scores,
#         'overall_readiness': int(overall_readiness),
#         'readiness_level': 'Professional' if overall_readiness > 85 else 
#                           'Advanced' if overall_readiness > 70 else
#                           'Intermediate' if overall_readiness > 55 else 'Beginner'
#     }

# # Helper functions for calculations
# def analyze_volume_trend(rms_db):
#     """Analyze if volume trends up, down, or stays consistent"""
#     if len(rms_db) < 10:
#         return "Insufficient data"
    
#     # Linear regression to find trend
#     x = np.arange(len(rms_db))
#     slope, _, r_value, _, _ = scipy.stats.linregress(x, rms_db)
    
#     if abs(slope) < 0.01:
#         return "Consistent"
#     elif slope > 0.01:
#         return "Building Energy"
#     else:
#         return "Fading Energy"

# def calculate_energy_sustainability(rms_db):
#     """Calculate how well energy is sustained throughout speech"""
#     if len(rms_db) < 10:
#         return 50
    
#     # Compare first and last thirds
#     first_third = np.mean(rms_db[:len(rms_db)//3])
#     last_third = np.mean(rms_db[-len(rms_db)//3:])
    
#     # Sustainability score (0-100)
#     energy_drop = first_third - last_third
#     sustainability = 100 - max(0, energy_drop * 10)  # Penalize energy drops
    
#     return max(0, min(100, sustainability))

# def calculate_rhythm_score(speech_activity, sr, hop_length):
#     """Calculate rhythm consistency score"""
#     # Convert to time domain
#     times = librosa.frames_to_time(np.arange(len(speech_activity)), sr=sr, hop_length=hop_length)
    
#     # Find speech segments
#     speech_changes = np.diff(speech_activity.astype(int))
#     speech_starts = times[:-1][speech_changes == 1]
#     speech_ends = times[:-1][speech_changes == -1]
    
#     if len(speech_starts) < 3:
#         return 50  # Not enough data
    
#     # Calculate segment durations
#     if len(speech_ends) < len(speech_starts):
#         speech_ends = np.append(speech_ends, times[-1])
    
#     segment_durations = speech_ends[:len(speech_starts)] - speech_starts
    
#     # Rhythm score based on consistency of segment lengths
#     duration_std = np.std(segment_durations)
#     duration_mean = np.mean(segment_durations)
    
#     if duration_mean == 0:
#         return 50
    
#     consistency = 100 - min(100, (duration_std / duration_mean) * 100)
#     return max(0, consistency)

# def estimate_speaking_pace(speech_activity, sr, hop_length):
#     """Estimate words per minute from speech patterns"""
#     # Rough estimation based on speech density and typical syllable rates
    
#     frame_duration = hop_length / sr
#     total_speech_time = np.sum(speech_activity) * frame_duration
#     total_time = len(speech_activity) * frame_duration
    
#     if total_speech_time == 0:
#         return 0
    
#     # Estimate based on speech density and typical patterns
#     # Average speaker: ~150 WPM, adjusted by speech density
#     speech_density = total_speech_time / total_time
#     estimated_wpm = 150 * speech_density * 1.2  # Adjustment factor
    
#     return min(300, max(50, estimated_wpm))  # Reasonable bounds

# def categorize_pace(wpm):
#     """Categorize speaking pace"""
#     if wpm < 100:
#         return "Very Slow"
#     elif wpm < 130:
#         return "Slow"
#     elif wpm < 160:
#         return "Optimal"
#     elif wpm < 200:
#         return "Fast"
#     else:
#         return "Very Fast"

# def detect_rushed_sections(speech_activity, sr, hop_length):
#     """Detect sections where speaking is rushed"""
#     # Simplified implementation - look for very dense speech regions
#     window_size = int(5 * sr / hop_length)  # 5-second windows
#     rushed_sections = []
    
#     for i in range(0, len(speech_activity) - window_size, window_size // 2):
#         window = speech_activity[i:i + window_size]
#         speech_density = np.mean(window)
        
#         if speech_density > 0.9:  # Very high speech density
#             rushed_sections.append(i)
    
#     return rushed_sections

# def calculate_rhythm_consistency(speech_activity):
#     """Calculate rhythm consistency score"""
#     # Look at speech pattern regularity
#     if len(speech_activity) < 100:
#         return 50
    
#     # Autocorrelation to find rhythmic patterns
#     autocorr = np.correlate(speech_activity, speech_activity, mode='full')
#     autocorr = autocorr[autocorr.size // 2:]
    
#     # Find peak in autocorrelation (indicates rhythmic consistency)
#     peaks, _ = find_peaks(autocorr[10:100])  # Look for patterns between 0.5-5 seconds
    
#     if len(peaks) == 0:
#         return 30  # No clear rhythm
    
#     # Score based on strongest rhythmic pattern
#     max_peak = np.max(autocorr[peaks + 10])
#     consistency = min(100, max_peak * 200)
    
#     return int(consistency)

# def find_pause_regions(is_silence, sr, hop_length):
#     """Find pause regions in speech"""
#     frame_duration = hop_length / sr
#     min_pause_duration = 0.3  # Minimum 300ms to be considered a pause
#     min_pause_frames = int(min_pause_duration / frame_duration)
    
#     pause_regions = []
#     in_pause = False
#     pause_start = 0
    
#     for i, silent in enumerate(is_silence):
#         if silent and not in_pause:
#             pause_start = i
#             in_pause = True
#         elif not silent and in_pause:
#             pause_duration = (i - pause_start) * frame_duration
#             if pause_duration >= min_pause_duration:
#                 pause_regions.append({
#                     'start': pause_start * frame_duration,
#                     'end': i * frame_duration,
#                     'duration': pause_duration
#                 })
#             in_pause = False
    
#     return pause_regions

# def calculate_pause_metrics(pause_regions, total_duration):
#     """Calculate pause-related metrics"""
#     if not pause_regions:
#         return {
#             'avg_duration': 0,
#             'frequency': 0,
#             'max_duration': 0,
#             'distribution': 'No pauses detected'
#         }
    
#     durations = [p['duration'] for p in pause_regions]
    
#     return {
#         'avg_duration': np.mean(durations),
#         'frequency': len(pause_regions) / (total_duration / 60),  # per minute
#         'max_duration': np.max(durations),
#         'distribution': categorize_pause_distribution(pause_regions, total_duration)
#     }

# def categorize_pause_distribution(pause_regions, total_duration):
#     """Categorize how pauses are distributed"""
#     if len(pause_regions) < 3:
#         return "Too few pauses"
    
#     # Analyze distribution across speech
#     thirds = total_duration / 3
#     first_third = sum(1 for p in pause_regions if p['start'] < thirds)
#     second_third = sum(1 for p in pause_regions if thirds <= p['start'] < 2 * thirds)
#     third_third = sum(1 for p in pause_regions if p['start'] >= 2 * thirds)
    
#     # Check if evenly distributed
#     total_pauses = len(pause_regions)
#     if max(first_third, second_third, third_third) < total_pauses * 0.6:
#         return "Well distributed"
#     elif first_third > total_pauses * 0.6:
#         return "Front-loaded"
#     elif third_third > total_pauses * 0.6:
#         return "End-loaded"
#     else:
#         return "Middle-heavy"

# def calculate_strategic_pause_score(pause_regions):
#     """Calculate how strategically pauses are used"""
#     if not pause_regions:
#         return 20  # Very low score for no pauses
    
#     # Ideal pause characteristics:
#     # - 3-8 pauses per minute
#     # - 0.5-2 second duration
#     # - Well distributed
    
#     durations = [p['duration'] for p in pause_regions]
#     avg_duration = np.mean(durations)
    
#     # Duration score (0-100)
#     if 0.5 <= avg_duration <= 2.0:
#         duration_score = 100
#     else:
#         duration_score = max(0, 100 - abs(avg_duration - 1.25) * 40)
    
#     # Frequency score (assuming ~3 minutes average speech)
#     ideal_pause_count = len(pause_regions)
#     if 3 <= ideal_pause_count <= 12:  # 1-4 pauses per minute
#         frequency_score = 100
#     else:
#         frequency_score = max(0, 100 - abs(ideal_pause_count - 7.5) * 10)
    
#     # Distribution score (simplified)
#     distribution_score = 80  # Default good score
    
#     # Combined strategic score
#     strategic_score = (duration_score * 0.4 + frequency_score * 0.4 + distribution_score * 0.2)
    
#     return int(strategic_score)


import librosa
import numpy as np
import scipy.stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from whisper_helper import robust_load

def analyze_vocal_dynamics(audio_path: str) -> Dict[str, Any]:
    """
    Advanced vocal-dynamics analysis for presentation coaching.
    Returns detailed metrics about pitch variation, volume, rhythm, and pauses.
    """
    try:
        # Always load with the unified loader ➜ handles m4a → wav, resampling, mono
        y, sr = robust_load(audio_path, sr=22_050)     # ← 🔧 changed line
        duration = len(y) / sr

        # 1. PITCH VARIATION ANALYSIS
        pitch_metrics = analyze_pitch_variation(y, sr)

        # 2. VOLUME DYNAMICS ANALYSIS
        volume_metrics = analyze_volume_dynamics(y, sr)

        # 3. SPEAKING RHYTHM ANALYSIS
        rhythm_metrics = analyze_speaking_rhythm(y, sr)

        # 4. STRATEGIC PAUSE ANALYSIS
        pause_metrics = analyze_pauses(y, sr)

        # 5. OVERALL VOCAL-DYNAMICS SCORE
        overall_score = calculate_vocal_dynamics_score(
            pitch_metrics, volume_metrics, rhythm_metrics, pause_metrics
        )

        return {
            "duration": duration,
            "pitch_dynamics": pitch_metrics,
            "volume_dynamics": volume_metrics,
            "rhythm_analysis": rhythm_metrics,
            "pause_analysis": pause_metrics,
            "overall_dynamics_score": overall_score,
            "vocal_health_indicators": analyze_vocal_health(y, sr),
            "presentation_readiness": calculate_presentation_readiness(
                pitch_metrics, volume_metrics, rhythm_metrics, pause_metrics
            ),
        }

    except Exception as e:
        print(f"[ERROR] Vocal dynamics analysis failed: {e}")
        return None

def analyze_pitch_variation(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze pitch variation and monotone detection with improved fundamental frequency detection"""
    
    # Use librosa.yin for more accurate fundamental frequency detection
    # YIN algorithm is specifically designed for pitch detection
    try:
        # Set frequency range for human speech
        fmin = 50   # Minimum human fundamental frequency  
        fmax = 400  # Maximum human fundamental frequency
        
        # Use YIN algorithm for better pitch detection
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
        
        # Filter out invalid pitches and silence
        valid_pitches = f0[(f0 > fmin) & (f0 < fmax) & ~np.isnan(f0)]
        
    except Exception as e:
        print(f"YIN algorithm failed, falling back to piptrack: {e}")
        # Fallback to improved piptrack method
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, 
                                             fmin=50, fmax=400)  # Constrain frequency range
        
        # Extract pitch values more carefully
        valid_pitches = []
        for t in range(pitches.shape[1]):
            # Get the strongest pitch in each frame
            mag_frame = magnitudes[:, t]
            pitch_frame = pitches[:, t]
            
            if mag_frame.max() > 0:
                # Find the index of the strongest magnitude
                max_mag_idx = mag_frame.argmax()
                pitch = pitch_frame[max_mag_idx]
                
                # Only accept pitches in human speech range
                if 50 <= pitch <= 400:
                    valid_pitches.append(pitch)
        
        valid_pitches = np.array(valid_pitches)
    
    if len(valid_pitches) < 10:  # Not enough data
        return {
            'variation_score': 0,
            'monotone_risk': 100,
            'pitch_range_hz': 0,
            'pitch_stability': 0,
            'dynamic_range': 'Very Limited',
            'mean_pitch': 0,
            'pitch_coefficient_variation': 0
        }
    
    # Calculate pitch variation metrics
    pitch_std = np.std(valid_pitches)
    pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
    pitch_mean = np.mean(valid_pitches)
    
    # Pitch variation score (0-100, higher = more dynamic)
    # Good speakers have 15-30% pitch variation
    variation_coefficient = pitch_std / pitch_mean if pitch_mean > 0 else 0
    variation_score = min(100, variation_coefficient * 300)  # Scale to 0-100
    
    # Monotone risk (0-100, higher = more monotone)
    monotone_risk = max(0, 100 - variation_score)
    
    # Pitch stability (consistency without being monotone)
    pitch_stability = 100 - (abs(variation_coefficient - 0.2) * 250)  # Optimal around 20%
    pitch_stability = max(0, min(100, pitch_stability))
    
    # Dynamic range classification
    if variation_score > 80:
        dynamic_range = "Highly Dynamic"
    elif variation_score > 60:
        dynamic_range = "Well Varied"
    elif variation_score > 40:
        dynamic_range = "Moderately Varied"
    elif variation_score > 20:
        dynamic_range = "Somewhat Monotone"
    else:
        dynamic_range = "Very Monotone"
    
    return {
        'variation_score': int(variation_score),
        'monotone_risk': int(monotone_risk),
        'pitch_range_hz': round(pitch_range, 1),
        'pitch_stability': int(pitch_stability),
        'dynamic_range': dynamic_range,
        'mean_pitch': round(pitch_mean, 1),
        'pitch_coefficient_variation': round(variation_coefficient * 100, 1)
    }

def analyze_volume_dynamics(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze volume/energy dynamics throughout speech"""
    
    # Calculate RMS energy in windows
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB for more meaningful analysis
    rms_db = 20 * np.log10(rms + 1e-6)  # Add small value to avoid log(0)
    
    # Volume metrics
    volume_range = np.max(rms_db) - np.min(rms_db)
    volume_std = np.std(rms_db)
    mean_volume = np.mean(rms_db)
    
    # Energy variation score
    energy_score = min(100, volume_std * 3)  # Scale to 0-100
    
    # Volume consistency (not too varied, not too flat)
    ideal_range = 12  # dB - good speakers have ~12dB dynamic range
    consistency_score = 100 - abs(volume_range - ideal_range) * 4
    consistency_score = max(0, min(100, consistency_score))
    
    # Detect volume patterns
    volume_trend = analyze_volume_trend(rms_db)
    
    # Energy sustainability (how well energy is maintained)
    sustainability = calculate_energy_sustainability(rms_db)
    
    return {
        'energy_score': int(energy_score),
        'volume_range_db': round(volume_range, 1),
        'consistency_score': int(consistency_score),
        'mean_volume_db': round(mean_volume, 1),
        'energy_sustainability': int(sustainability),
        'volume_trend': volume_trend,
        'dynamic_presence': 'Strong' if energy_score > 70 else 'Moderate' if energy_score > 40 else 'Weak'
    }

def analyze_speaking_rhythm(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze speaking rhythm and tempo variations"""
    
    # Detect speech segments using spectral features
    hop_length = 512
    
    # Use spectral centroid and rolloff to identify speech vs silence
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Combine features to detect speech activity
    speech_activity = (spectral_centroids > np.percentile(spectral_centroids, 25)) & \
                     (spectral_rolloff > np.percentile(spectral_rolloff, 25))
    
    # Calculate tempo and rhythm metrics
    rhythm_score = calculate_rhythm_score(speech_activity, sr, hop_length)
    
    # Pace analysis (words per minute estimation from audio features)
    estimated_pace = estimate_speaking_pace(speech_activity, sr, hop_length)
    
    # Rush detection (identifying sections that are too fast)
    rush_sections = detect_rushed_sections(speech_activity, sr, hop_length)
    
    return {
        'rhythm_score': int(rhythm_score),
        'estimated_pace_wpm': int(estimated_pace),
        'pace_category': categorize_pace(estimated_pace),
        'rush_risk': int(len(rush_sections) * 10),  # Higher if more rushed sections
        'rhythm_consistency': calculate_rhythm_consistency(speech_activity),
        'speech_timing': 'Excellent' if rhythm_score > 80 else 'Good' if rhythm_score > 60 else 'Needs Work'
    }

def analyze_pauses(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze strategic pause placement and effectiveness"""
    
    # Detect silence/pause regions
    frame_length = 2048
    hop_length = 512
    
    # Use RMS energy to detect pauses
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Threshold for silence (adaptive based on audio)
    silence_threshold = np.percentile(rms, 20)  # Bottom 20% considered silence
    
    # Find pause regions
    is_silence = rms < silence_threshold
    pause_regions = find_pause_regions(is_silence, sr, hop_length)
    
    # Analyze pause characteristics
    pause_metrics = calculate_pause_metrics(pause_regions, len(y) / sr)
    
    # Strategic pause score (based on placement and duration)
    strategic_score = calculate_strategic_pause_score(pause_regions)
    
    return {
        'total_pauses': len(pause_regions),
        'average_pause_duration': round(pause_metrics['avg_duration'], 2),
        'pause_frequency': round(pause_metrics['frequency'], 2),  # pauses per minute
        'strategic_score': int(strategic_score),
        'pause_effectiveness': 'Excellent' if strategic_score > 80 else 'Good' if strategic_score > 60 else 'Needs Improvement',
        'longest_pause': round(pause_metrics['max_duration'], 2),
        'pause_distribution': pause_metrics['distribution']
    }

def calculate_vocal_dynamics_score(pitch_metrics: Dict, volume_metrics: Dict, 
                                 rhythm_metrics: Dict, pause_metrics: Dict) -> int:
    """Calculate overall vocal dynamics score"""
    
    # Weight different components
    weights = {
        'pitch': 0.3,
        'volume': 0.25, 
        'rhythm': 0.25,
        'pauses': 0.2
    }
    
    pitch_score = pitch_metrics['variation_score']
    volume_score = volume_metrics['consistency_score']
    rhythm_score = rhythm_metrics['rhythm_score']
    pause_score = pause_metrics['strategic_score']
    
    overall = (pitch_score * weights['pitch'] + 
              volume_score * weights['volume'] + 
              rhythm_score * weights['rhythm'] + 
              pause_score * weights['pauses'])
    
    return int(overall)

def analyze_vocal_health(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Analyze vocal health indicators"""
    
    # Spectral features that indicate vocal strain or health
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Voice quality indicators
    breathiness = np.mean(zero_crossing_rate) * 1000  # Higher = more breathy
    clarity = np.mean(spectral_centroids) / 1000  # Voice clarity indicator
    
    return {
        'vocal_clarity': min(100, int(clarity * 20)),
        'breathiness_level': min(100, int(breathiness * 50)),
        'vocal_strain_risk': 'Low' if breathiness < 0.1 else 'Moderate' if breathiness < 0.2 else 'High'
    }

def calculate_presentation_readiness(pitch_metrics: Dict, volume_metrics: Dict, 
                                   rhythm_metrics: Dict, pause_metrics: Dict) -> Dict[str, Any]:
    """Calculate overall presentation readiness metrics"""
    
    # Professional presentation benchmarks
    benchmarks = {
        'pitch_variation': 60,  # Good speakers have 60+ variation score
        'volume_consistency': 70,  # Should have good volume control
        'rhythm_score': 65,  # Steady rhythm
        'pause_effectiveness': 60  # Strategic pauses
    }
    
    readiness_scores = {
        'pitch_readiness': min(100, (pitch_metrics['variation_score'] / benchmarks['pitch_variation']) * 100),
        'volume_readiness': min(100, (volume_metrics['consistency_score'] / benchmarks['volume_consistency']) * 100),
        'rhythm_readiness': min(100, (rhythm_metrics['rhythm_score'] / benchmarks['rhythm_score']) * 100),
        'pause_readiness': min(100, (pause_metrics['strategic_score'] / benchmarks['pause_effectiveness']) * 100)
    }
    
    overall_readiness = sum(readiness_scores.values()) / len(readiness_scores)
    
    return {
        **readiness_scores,
        'overall_readiness': int(overall_readiness),
        'readiness_level': 'Professional' if overall_readiness > 85 else 
                          'Advanced' if overall_readiness > 70 else
                          'Intermediate' if overall_readiness > 55 else 'Beginner'
    }

# Helper functions for calculations
def analyze_volume_trend(rms_db):
    """Analyze if volume trends up, down, or stays consistent"""
    if len(rms_db) < 10:
        return "Insufficient data"
    
    # Linear regression to find trend
    x = np.arange(len(rms_db))
    slope, _, r_value, _, _ = scipy.stats.linregress(x, rms_db)
    
    if abs(slope) < 0.01:
        return "Consistent"
    elif slope > 0.01:
        return "Building Energy"
    else:
        return "Fading Energy"

def calculate_energy_sustainability(rms_db):
    """Calculate how well energy is sustained throughout speech"""
    if len(rms_db) < 10:
        return 50
    
    # Compare first and last thirds
    first_third = np.mean(rms_db[:len(rms_db)//3])
    last_third = np.mean(rms_db[-len(rms_db)//3:])
    
    # Sustainability score (0-100)
    energy_drop = first_third - last_third
    sustainability = 100 - max(0, energy_drop * 10)  # Penalize energy drops
    
    return max(0, min(100, sustainability))

def calculate_rhythm_score(speech_activity, sr, hop_length):
    """Calculate rhythm consistency score"""
    # Convert to time domain
    times = librosa.frames_to_time(np.arange(len(speech_activity)), sr=sr, hop_length=hop_length)
    
    # Find speech segments
    speech_changes = np.diff(speech_activity.astype(int))
    speech_starts = times[:-1][speech_changes == 1]
    speech_ends = times[:-1][speech_changes == -1]
    
    if len(speech_starts) < 3:
        return 50  # Not enough data
    
    # Calculate segment durations
    if len(speech_ends) < len(speech_starts):
        speech_ends = np.append(speech_ends, times[-1])
    
    segment_durations = speech_ends[:len(speech_starts)] - speech_starts
    
    # Rhythm score based on consistency of segment lengths
    duration_std = np.std(segment_durations)
    duration_mean = np.mean(segment_durations)
    
    if duration_mean == 0:
        return 50
    
    consistency = 100 - min(100, (duration_std / duration_mean) * 100)
    return max(0, consistency)

def estimate_speaking_pace(speech_activity, sr, hop_length):
    """Estimate words per minute from speech patterns"""
    # Rough estimation based on speech density and typical syllable rates
    
    frame_duration = hop_length / sr
    total_speech_time = np.sum(speech_activity) * frame_duration
    total_time = len(speech_activity) * frame_duration
    
    if total_speech_time == 0:
        return 0
    
    # Estimate based on speech density and typical patterns
    # Average speaker: ~150 WPM, adjusted by speech density
    speech_density = total_speech_time / total_time
    estimated_wpm = 150 * speech_density * 1.2  # Adjustment factor
    
    return min(300, max(50, estimated_wpm))  # Reasonable bounds

def categorize_pace(wpm):
    """Categorize speaking pace"""
    if wpm < 100:
        return "Very Slow"
    elif wpm < 130:
        return "Slow"
    elif wpm < 160:
        return "Optimal"
    elif wpm < 200:
        return "Fast"
    else:
        return "Very Fast"

def detect_rushed_sections(speech_activity, sr, hop_length):
    """Detect sections where speaking is rushed"""
    # Simplified implementation - look for very dense speech regions
    window_size = int(5 * sr / hop_length)  # 5-second windows
    rushed_sections = []
    
    for i in range(0, len(speech_activity) - window_size, window_size // 2):
        window = speech_activity[i:i + window_size]
        speech_density = np.mean(window)
        
        if speech_density > 0.9:  # Very high speech density
            rushed_sections.append(i)
    
    return rushed_sections

def calculate_rhythm_consistency(speech_activity):
    """Calculate rhythm consistency score"""
    # Look at speech pattern regularity
    if len(speech_activity) < 100:
        return 50
    
    # Autocorrelation to find rhythmic patterns
    autocorr = np.correlate(speech_activity, speech_activity, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # Find peak in autocorrelation (indicates rhythmic consistency)
    peaks, _ = find_peaks(autocorr[10:100])  # Look for patterns between 0.5-5 seconds
    
    if len(peaks) == 0:
        return 30  # No clear rhythm
    
    # Score based on strongest rhythmic pattern
    max_peak = np.max(autocorr[peaks + 10])
    consistency = min(100, max_peak * 200)
    
    return int(consistency)

def find_pause_regions(is_silence, sr, hop_length):
    """Find pause regions in speech"""
    frame_duration = hop_length / sr
    min_pause_duration = 0.3  # Minimum 300ms to be considered a pause
    min_pause_frames = int(min_pause_duration / frame_duration)
    
    pause_regions = []
    in_pause = False
    pause_start = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_pause:
            pause_start = i
            in_pause = True
        elif not silent and in_pause:
            pause_duration = (i - pause_start) * frame_duration
            if pause_duration >= min_pause_duration:
                pause_regions.append({
                    'start': pause_start * frame_duration,
                    'end': i * frame_duration,
                    'duration': pause_duration
                })
            in_pause = False
    
    return pause_regions

def calculate_pause_metrics(pause_regions, total_duration):
    """Calculate pause-related metrics"""
    if not pause_regions:
        return {
            'avg_duration': 0,
            'frequency': 0,
            'max_duration': 0,
            'distribution': 'No pauses detected'
        }
    
    durations = [p['duration'] for p in pause_regions]
    
    return {
        'avg_duration': np.mean(durations),
        'frequency': len(pause_regions) / (total_duration / 60),  # per minute
        'max_duration': np.max(durations),
        'distribution': categorize_pause_distribution(pause_regions, total_duration)
    }

def categorize_pause_distribution(pause_regions, total_duration):
    """Categorize how pauses are distributed"""
    if len(pause_regions) < 3:
        return "Too few pauses"
    
    # Analyze distribution across speech
    thirds = total_duration / 3
    first_third = sum(1 for p in pause_regions if p['start'] < thirds)
    second_third = sum(1 for p in pause_regions if thirds <= p['start'] < 2 * thirds)
    third_third = sum(1 for p in pause_regions if p['start'] >= 2 * thirds)
    
    # Check if evenly distributed
    total_pauses = len(pause_regions)
    if max(first_third, second_third, third_third) < total_pauses * 0.6:
        return "Well distributed"
    elif first_third > total_pauses * 0.6:
        return "Front-loaded"
    elif third_third > total_pauses * 0.6:
        return "End-loaded"
    else:
        return "Middle-heavy"

def calculate_strategic_pause_score(pause_regions):
    """Calculate how strategically pauses are used"""
    if not pause_regions:
        return 20  # Very low score for no pauses
    
    # Ideal pause characteristics:
    # - 3-8 pauses per minute
    # - 0.5-2 second duration
    # - Well distributed
    
    durations = [p['duration'] for p in pause_regions]
    avg_duration = np.mean(durations)
    
    # Duration score (0-100)
    if 0.5 <= avg_duration <= 2.0:
        duration_score = 100
    else:
        duration_score = max(0, 100 - abs(avg_duration - 1.25) * 40)
    
    # Frequency score (assuming ~3 minutes average speech)
    ideal_pause_count = len(pause_regions)
    if 3 <= ideal_pause_count <= 12:  # 1-4 pauses per minute
        frequency_score = 100
    else:
        frequency_score = max(0, 100 - abs(ideal_pause_count - 7.5) * 10)
    
    # Distribution score (simplified)
    distribution_score = 80  # Default good score
    
    # Combined strategic score
    strategic_score = (duration_score * 0.4 + frequency_score * 0.4 + distribution_score * 0.2)
    
    return int(strategic_score)