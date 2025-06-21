import requests
import json
import base64
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("TTS_API_KEY")
API_URL = os.getenv("TTS_URL")

AUDIO_FILES = os.path.join(os.getcwd(), 'static', 'generated_audio')

def get_speech(text, gender='Female', filename='speech_'):
    try:
        # Ensure the directory exists
        os.makedirs(AUDIO_FILES, exist_ok=True)
        
        if gender == 'Female':
            voice = 'sage'
        elif gender == 'Male':
            voice = 'onyx'
        else:
            voice = 'echo'

        # Payload
        headers = {
            "api-key": API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            "voice": voice,
            "model": "gpt-4o-mini-tts"
        }

        suffix = int(datetime.now().timestamp())
        final_filename = f"{filename}_{suffix}"
        full_path = os.path.join(AUDIO_FILES, f"{final_filename}.mp3")
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            with open(full_path, 'wb') as f:
                f.write(response.content)
            return True, f'{final_filename}.mp3'
        else:

            error_msg = f"HTTP {response.status_code}: {response.text}"
            return False, error_msg
            
    except Exception as e:
        return False, str(e)

# # Test the function
# print(int(datetime.now().timestamp()))
# success, result = get_speech("Hello! This is a sample using GPT-4o Mini for text-to-speech.", gender='Female', filename='speech_')

# if success:
#     print(f"Success! Audio file created: {result}")
# else:
#     print(f"Error: {result}")