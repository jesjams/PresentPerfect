import json, re, os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

load_dotenv()

USE_AZURE = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
MODEL_NAME = os.getenv("OPENAI_MODEL")

# Initialize OpenAI or AzureOpenAI client
if USE_AZURE:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21"
    )
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_transcript_for_presentation(original_transcript: str):
    """Create enhanced transcript optimized for presentations"""

    system_prompt = """You are a speaking coach. Transform this rough transcript of an audio of someone speaking into a polished, extremely high-quality script. Wherever required, add punctuations. Join sentences perfectly. 
Ensure there's a smooth transition at all times. This is going to be used as a teaching script, so a high quality output is required.

Rules:
1. Remove filler words
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
        completion = client.chat.completions.create(
            model=MODEL_NAME,
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
