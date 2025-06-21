import os
import re
import boto3
import tempfile
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from openai import OpenAI
import pickle

class PresentationAudioMetrics(BaseModel):
    clarity_score: int  # 1-100, how clear and well-structured the speech is
    pace_score: int     # 1-100, speaking pace appropriateness
    confidence_score: int  # 1-100, confidence indicators (fewer fillers, good pacing)
    engagement_score: int  # 1-100, how engaging the content is
    overall_score: int     # 1-100, average of above scores
    
    # Actionable feedback
    clarity_feedback: str
    pace_feedback: str
    confidence_feedback: str
    engagement_feedback: str
    
    # Key improvements made
    improvements_summary: str
    key_changes: List[str]
    speaking_tips: List[str]

class EnhancedAudioProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.polly_client = boto3.client(
            'polly',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        # Create cache directory
        self.cache_dir = os.path.join(tempfile.gettempdir(), 'audio_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for caching"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_cached_result(self, file_hash: str) -> Optional[Dict]:
        """Check if we have cached results for this file"""
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def save_to_cache(self, file_hash: str, result: Dict):
        """Save results to cache"""
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"[WARNING] Failed to cache result: {e}")
    
    def analyze_presentation_quality(self, transcript: str, speech_metrics: Dict) -> PresentationAudioMetrics:
        """Analyze transcript for presentation-specific improvements"""
        
        system_prompt = """You are an expert presentation coach. Analyze this speech transcript and provide specific, actionable feedback for presentation improvement.

Focus on these key areas:
1. CLARITY: Is the message clear, well-structured, and easy to follow?
2. PACE: Is the speaking pace appropriate for audience comprehension?
3. CONFIDENCE: Does the speaker sound confident? Look for filler words, hesitations.
4. ENGAGEMENT: Is the content engaging and well-delivered?

Provide scores (1-100) and specific, actionable feedback for each area.
Return JSON with exactly these fields: clarity_score, pace_score, confidence_score, engagement_score, overall_score, clarity_feedback, pace_feedback, confidence_feedback, engagement_feedback, improvements_summary, key_changes (array), speaking_tips (array of 3-4 tips)."""

        user_prompt = f"""
TRANSCRIPT TO ANALYZE:
{transcript}

SPEECH METRICS:
- Speaking rate: {speech_metrics.get('speaking_rate', 0)} WPM
- Filler word ratio: {speech_metrics.get('filler_ratio', 0)}%
- Average pause length: {speech_metrics.get('avg_pause_length', 0)}s
- Word count: {speech_metrics.get('word_count', 0)}

Provide presentation-focused analysis and actionable improvement suggestions."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PresentationAudioMetrics,
                temperature=0.3
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze presentation quality: {e}")
            return self._create_fallback_metrics(transcript, speech_metrics)
    
    def _create_fallback_metrics(self, transcript: str, speech_metrics: Dict) -> PresentationAudioMetrics:
        """Create fallback metrics if API call fails"""
        words = transcript.split()
        word_count = len(words)
        
        # Basic scoring based on metrics
        pace_score = 85 if 120 <= speech_metrics.get('speaking_rate', 0) <= 160 else 60
        confidence_score = 90 if speech_metrics.get('filler_ratio', 0) < 2 else 70
        clarity_score = 80 if word_count > 50 else 60
        engagement_score = 75  # Default
        
        overall_score = (pace_score + confidence_score + clarity_score + engagement_score) // 4
        
        return PresentationAudioMetrics(
            clarity_score=clarity_score,
            pace_score=pace_score,
            confidence_score=confidence_score,
            engagement_score=engagement_score,
            overall_score=overall_score,
            clarity_feedback="Focus on structuring your message with clear beginning, middle, and end.",
            pace_feedback="Practice maintaining a steady, comfortable speaking pace.",
            confidence_feedback="Work on reducing filler words and speaking with conviction.",
            engagement_feedback="Add more energy and enthusiasm to capture audience attention.",
            improvements_summary="Basic presentation improvements needed.",
            key_changes=["Reduce filler words", "Improve pacing", "Add structure"],
            speaking_tips=[
                "Practice your opening and closing statements",
                "Use pause instead of filler words",
                "Record yourself regularly to track progress",
                "Focus on one key message per sentence"
            ]
        )
    
    def enhance_transcript_for_presentation(self, original_transcript: str) -> Dict[str, Any]:
        """Create enhanced transcript optimized for presentations"""
        
        system_prompt = """You are a presentation coach. Transform this rough transcript into a polished, presentation-ready script.

Rules:
1. Remove filler words (um, uh, like, etc.)
2. Fix grammar and sentence structure
3. Add clear transitions between ideas
4. Make it sound confident and engaging
5. Keep the speaker's original message and style
6. Make it suitable for spoken delivery (not written)

Return JSON with:
- enhanced_text: The improved transcript
- improvements_made: List of specific improvements
- presentation_tips: 3-4 actionable speaking tips"""

        user_prompt = f"""Original transcript to enhance:

{original_transcript}

Transform this into a confident, well-structured presentation script that sounds natural when spoken aloud."""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(completion.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to enhance transcript: {e}")
            # Fallback: basic cleanup
            cleaned = re.sub(r'\b(um|uh|like|you know|basically|actually)\b', '', original_transcript, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return {
                "enhanced_text": cleaned,
                "improvements_made": ["Removed filler words", "Basic cleanup applied"],
                "presentation_tips": [
                    "Practice reducing filler words",
                    "Speak with confidence and clarity",
                    "Use strategic pauses for emphasis"
                ]
            }
    
    def create_simple_ssml(self, text: str) -> str:
        """Create simple, reliable SSML for AWS Polly"""
        # Clean the text
        clean_text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Add basic pauses after sentences and clauses
        text_with_pauses = re.sub(r'([.!?])\s+', r'\1<break time="750ms"/> ', clean_text)
        text_with_pauses = re.sub(r'([,;:])\s+', r'\1<break time="300ms"/> ', text_with_pauses)
        
        # Create simple SSML
        ssml = f'''<speak>
<prosody rate="medium" pitch="medium">
<break time="500ms"/>
{text_with_pauses}
<break time="500ms"/>
</prosody>
</speak>'''
        
        return ssml
    
    def generate_enhanced_audio(self, enhanced_text: str, session_id: str) -> Optional[str]:
        """Generate enhanced audio using AWS Polly with timestamp"""
        try:
            # Create simple SSML
            ssml_content = self.create_simple_ssml(enhanced_text)
            
            # Use timestamp for unique filename
            timestamp = int(time.time())
            filename = f"enhanced_{session_id}_{timestamp}.mp3"
            
            # Use absolute path to ensure it's saved in the right place
            audio_dir = os.path.join(os.getcwd(), 'static', 'generated_audio')
            os.makedirs(audio_dir, exist_ok=True)
            
            output_path = os.path.join(audio_dir, filename)
            
            print(f"[INFO] Generating enhanced audio: {filename}")
            print(f"[INFO] Saving to absolute path: {output_path}")
            
            # Try neural engine first, fallback to standard if not supported
            try:
                response = self.polly_client.synthesize_speech(
                    Text=ssml_content,
                    TextType='ssml',
                    OutputFormat='mp3',
                    VoiceId='Joanna',  # Clear, professional US English voice
                    Engine='neural',   # Try neural first
                    SampleRate='24000'
                )
            except Exception as neural_error:
                print(f"[WARNING] Neural engine failed: {neural_error}")
                print("[INFO] Falling back to standard engine...")
                
                # Fallback to standard engine
                response = self.polly_client.synthesize_speech(
                    Text=ssml_content,
                    TextType='ssml',
                    OutputFormat='mp3',
                    VoiceId='Joanna',  # Same voice, standard engine
                    Engine='standard', # Standard engine (more widely supported)
                    SampleRate='22050' # Standard sample rate
                )
            
            # Save audio file
            with open(output_path, 'wb') as file:
                file.write(response['AudioStream'].read())
            
            print(f"[INFO] Enhanced audio saved: {output_path}")
            print(f"[INFO] File exists after save: {os.path.exists(output_path)}")
            print(f"[INFO] File size: {os.path.getsize(output_path) if os.path.exists(output_path) else 'N/A'} bytes")
            
            # Return the relative path for the URL
            return os.path.join('static', 'generated_audio', filename)
            
        except Exception as e:
            print(f"[ERROR] Failed to generate enhanced audio: {e}")
            import traceback
            traceback.print_exc()
            return None

def process_audio_for_presentation(temp_path, processor=None, socketio_instance=None):
    """
    Process audio file for presentation improvement
    """
    if processor is None:
        processor = EnhancedAudioProcessor()
    
    try:
        # Check cache first
        file_hash = processor.get_file_hash(temp_path)
        cached_result = processor.get_cached_result(file_hash)
        
        if cached_result:
            print("[INFO] Using cached transcription result")
            if socketio_instance:
                socketio_instance.emit('processing-update', 
                    {'message': 'Found cached analysis, enhancing...', 'progress': 40})
        else:
            # Step 1: Transcription
            if socketio_instance:
                socketio_instance.emit('processing-update', 
                    {'message': 'Transcribing your audio... 🎵', 'progress': 20})
            
            import whisper
            whisper_model = whisper.load_model("turbo", device="cpu")
            
            result = whisper_model.transcribe(temp_path, fp16=False, verbose=False)
            segments = result.get("segments", [])
            full_text = result.get("text", "").strip()
            language = result.get("language", "unknown")
            duration = max(seg.get('end', 0) for seg in segments) if segments else 0
            
            # Cache the transcription result
            cached_result = {
                'segments': segments,
                'full_text': full_text,
                'language': language,
                'duration': duration
            }
            processor.save_to_cache(file_hash, cached_result)
        
        # Extract cached data
        segments = cached_result['segments']
        full_text = cached_result['full_text']
        language = cached_result['language']
        duration = cached_result['duration']
        
        print(f"[INFO] Transcription: {len(full_text)} characters, {duration:.1f}s")
        
        # Step 2: Analyze speech patterns
        if socketio_instance:
            socketio_instance.emit('processing-update', 
                {'message': 'Analyzing speech patterns... 📊', 'progress': 50})
        
        speech_analysis = analyze_speech_patterns_simple(segments, full_text, duration)
        
        # Step 3: Get presentation-focused metrics
        if socketio_instance:
            socketio_instance.emit('processing-update', 
                {'message': 'Analyzing presentation quality... 🎭', 'progress': 70})
        
        presentation_metrics = processor.analyze_presentation_quality(full_text, speech_analysis)
        
        # Step 4: Enhance transcript
        if socketio_instance:
            socketio_instance.emit('processing-update', 
                {'message': 'Creating enhanced version... ✨', 'progress': 85})
        
        enhancement_result = processor.enhance_transcript_for_presentation(full_text)
        enhanced_text = enhancement_result.get('enhanced_text', full_text)
        
        # Step 5: Generate enhanced audio
        session_id = file_hash[:8]  # Use part of hash as session ID
        enhanced_audio_path = processor.generate_enhanced_audio(enhanced_text, session_id)
        
        # Format transcript segments
        formatted_segments = "\n".join([
            f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}"
            for seg in segments
        ]) if segments else full_text
        
        # Prepare response
        payload = {
            'type': 'audio',
            'transcriptSegments': formatted_segments,
            'enhancedTranscript': enhanced_text,
            'enhancedAudioPath': enhanced_audio_path,
            'enhancedAudioUrl': f"/static/generated_audio/{os.path.basename(enhanced_audio_path)}" if enhanced_audio_path else None,
            'fullText': full_text,
            'segments': segments,
            'duration': duration,
            'language': language,
            
            # Presentation-focused metrics
            'presentationMetrics': {
                'clarity_score': presentation_metrics.clarity_score,
                'pace_score': presentation_metrics.pace_score,
                'confidence_score': presentation_metrics.confidence_score,
                'engagement_score': presentation_metrics.engagement_score,
                'overall_score': presentation_metrics.overall_score,
                'clarity_feedback': presentation_metrics.clarity_feedback,
                'pace_feedback': presentation_metrics.pace_feedback,
                'confidence_feedback': presentation_metrics.confidence_feedback,
                'engagement_feedback': presentation_metrics.engagement_feedback,
            },
            
            # Enhancement details
            'enhancement': {
                'summary': presentation_metrics.improvements_summary,
                'key_changes': presentation_metrics.key_changes,
                'speaking_tips': presentation_metrics.speaking_tips,
                'improvements_made': enhancement_result.get('improvements_made', [])
            },
            
            # Basic speech analysis (for compatibility)
            'speechAnalysis': speech_analysis,
            'overallScore': presentation_metrics.overall_score,
            'overallSummary': f"Presentation analysis complete! Overall score: {presentation_metrics.overall_score}/100. {presentation_metrics.improvements_summary}"
        }
        
        if socketio_instance:
            socketio_instance.emit('processing-complete',
                 {'message': 'Presentation analysis complete! 🎉', 'progress': 100, 'data': payload})
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"[ERROR] Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        if socketio_instance:
            socketio_instance.emit('processing-error', 
                {'error': f'Failed to process audio: {str(e)}'})

def analyze_speech_patterns_simple(segments, full_text, duration):
    """Simplified speech pattern analysis focused on presentation skills"""
    
    if not full_text.strip():
        return {
            'word_count': 0, 'speaking_rate': 0, 'filler_words': 0, 'filler_ratio': 0,
            'avg_pause_length': 0, 'long_pauses': 0, 'confidence_score': 50,
            'pace_feedback': 'No speech detected', 'filler_feedback': 'No speech detected'
        }
    
    words = full_text.split()
    word_count = len(words)
    speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
    
    # Count filler words
    filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally', 'so']
    filler_count = sum(full_text.lower().count(filler) for filler in filler_words)
    filler_ratio = (filler_count / word_count * 100) if word_count > 0 else 0
    
    # Analyze pauses
    pauses = []
    if len(segments) > 1:
        for i in range(1, len(segments)):
            gap = segments[i]['start'] - segments[i-1]['end']
            if gap > 0.5:
                pauses.append(gap)
    
    avg_pause = sum(pauses) / len(pauses) if pauses else 0
    long_pauses = len([p for p in pauses if p > 2.0])
    
    # Calculate confidence score
    confidence_score = 75  # Base score
    
    if 120 <= speaking_rate <= 160:
        confidence_score += 15
    elif speaking_rate < 90 or speaking_rate > 180:
        confidence_score -= 15
    
    if filler_ratio < 2:
        confidence_score += 10
    elif filler_ratio > 5:
        confidence_score -= 20
    
    confidence_score = max(0, min(100, confidence_score))
    
    return {
        'word_count': word_count,
        'speaking_rate': round(speaking_rate, 1),
        'filler_words': filler_count,
        'filler_ratio': round(filler_ratio, 1),
        'avg_pause_length': round(avg_pause, 2),
        'long_pauses': long_pauses,
        'confidence_score': confidence_score,
        'pace_feedback': get_pace_feedback_simple(speaking_rate),
        'filler_feedback': get_filler_feedback_simple(filler_ratio)
    }

def get_pace_feedback_simple(speaking_rate):
    if speaking_rate < 100:
        return "Try speaking a bit faster to maintain audience engagement."
    elif speaking_rate <= 160:
        return "Excellent speaking pace for presentations!"
    else:
        return "Slow down slightly to ensure your audience can follow along."

def get_filler_feedback_simple(filler_ratio):
    if filler_ratio < 2:
        return "Excellent! Very few filler words - sounds very professional."
    elif filler_ratio < 5:
        return "Good control of filler words, with room for minor improvement."
    else:
        return "Focus on reducing filler words - pause instead of saying 'um' or 'uh'."