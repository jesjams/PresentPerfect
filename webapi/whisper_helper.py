import whisper



def whisper_transcriber(path):
    try:
        whisper_model = whisper.load_model("turbo", device="cpu")        
        result = whisper_model.transcribe(path, fp16=False, verbose=False)
        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        duration = max(seg.get('end', 0) for seg in segments) if segments else 0
        data = {
            'segments': segments,
            'full_text': full_text,
            'language': language,
            'duration': duration
        }
        return data
    except Exception as e:
        print(f"[ERROR] Failed to transcribe audio: {e}")
        return None
        