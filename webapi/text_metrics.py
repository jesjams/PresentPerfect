import os, sys, wave, subprocess, json, time, string, difflib
from typing import List, Dict, Any, Optional
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

SPEECH_KEY       =  os.getenv("SPEECH_KEY")
SERVICE_REGION   = os.getenv("SERVICE_REGION")
LANGUAGE         = os.getenv("LANGUAGE")
SEGMENT_SILENCE_MS = int(os.getenv("SEGMENT_SILENCE_MS"))
MAX_WAIT_SEC       = int(os.getenv("MAX_WAIT_SEC"))
 
REFERENCE_TEXT = ""     # "" = unscripted
ENABLE_MISCUE  = True   # only matters if REFERENCE_TEXT is set

# Helper methods

def ensure_wav(src: str) -> str:
    base, ext = os.path.splitext(src)
    wav = base + ".wav"
    if ext.lower() != ".wav" or not os.path.exists(wav):
        print(f"[Info] Converting {src} → {wav}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", wav],
            check=True)
    return wav
 
def validate_wav(path: str) -> None:
    with wave.open(path, "rb") as wf:
        ch, wd, sr, nfrm, ctype, _ = wf.getparams()
        print(f"[OK] WAV: {ch}-ch, {wd*8}-bit, {sr} Hz, {nfrm/sr:.1f} s")
        if (ch, wd, sr, ctype) != (1, 2, 16000, "NONE"):
            print("[Warning] Not 16-kHz 16-bit mono PCM (SDK will resample).")


# Recognition function
def run_assessment(wav: str) -> List[Dict[str, Any]]:
    sc = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        endpoint=f"https://{SERVICE_REGION}.api.cognitive.microsoft.com")
    sc.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
        str(SEGMENT_SILENCE_MS))
 
    pa = speechsdk.PronunciationAssessmentConfig(
        reference_text=REFERENCE_TEXT,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=ENABLE_MISCUE)
    pa.enable_prosody_assessment()
 
    rec = speechsdk.SpeechRecognizer(
        speech_config=sc, language=LANGUAGE,
        audio_config=speechsdk.audio.AudioConfig(filename=wav))
    pa.apply_to(rec)
 
    raw: List[str] = []
    done = False
 
    def recog_cb(e):
        j = e.result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        if j: raw.append(j)
 
    def stop_cb(_):
        nonlocal done
        done = True
 
    rec.recognized.connect(recog_cb)
    rec.session_stopped.connect(stop_cb)
    rec.canceled.connect(stop_cb)
 
    rec.start_continuous_recognition()
    t0 = time.time()
    while not done and time.time() - t0 < MAX_WAIT_SEC:
        time.sleep(0.2)
    rec.stop_continuous_recognition()
    return [json.loads(j) for j in raw]
 
# Parsing + Scoring
def parse_segment(js: Dict[str, Any]) -> Dict[str, Any]:
    best = js["NBest"][0]
    pa   = best["PronunciationAssessment"]
    return {
        "Accuracy": pa["AccuracyScore"],
        "Fluency":  pa["FluencyScore"],
        "Prosody":  pa["ProsodyScore"],
        "Pron":     pa["PronScore"],
        "Words": [{
            "Word": w["Word"],
            "Duration": int(w["Duration"]),
            "Accuracy": w["PronunciationAssessment"]["AccuracyScore"],
            "ErrorType": w["PronunciationAssessment"].get("ErrorType","None")
        } for w in best["Words"]]
    }
 
def diff_miscue(ref: List[str], words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sm = difflib.SequenceMatcher(None, ref, [w["Word"].lower() for w in words])
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag in ("insert","replace"):
            for w in words[j1:j2]:
                w2 = w.copy()
                if w2["ErrorType"]=="None": w2["ErrorType"]="Insertion"
                out.append(w2)
        if tag in ("delete","replace"):
            for t in ref[i1:i2]:
                out.append({"Word":t,"Duration":0,"Accuracy":0.0,"ErrorType":"Omission"})
        if tag=="equal": out.extend(words[j1:j2])
    return out
 
def aggregate(segs: List[Dict[str, Any]]) -> Dict[str, Any]:
    words = [w for s in segs for w in s["Words"]]
    if REFERENCE_TEXT and ENABLE_MISCUE:
        ref = [w.strip(string.punctuation) for w in REFERENCE_TEXT.lower().split()]
        words = diff_miscue(ref, words)
 
    acc_vals = [w["Accuracy"] for w in words if w["ErrorType"]!="Insertion"]
    accuracy = sum(acc_vals)/len(acc_vals) if acc_vals else 0.0
 
    seg_dur = [sum(w["Duration"] for w in s["Words"]) for s in segs]
    seg_flu = [s["Fluency"] for s in segs]
    fluency = sum(f*d for f,d in zip(seg_flu,seg_dur))/sum(seg_dur) if sum(seg_dur) else 0.0
 
    prosody = sum(s["Prosody"] for s in segs)/len(segs)
 
    comp: Optional[float] = None
    if REFERENCE_TEXT:
        comp = (len([w for w in words if w["ErrorType"]=="None"]) /
                len(REFERENCE_TEXT.split()))*100
        comp = min(comp,100.0)
 
    pron = (0.4*accuracy+0.2*prosody+0.2*fluency+0.2*comp) if comp is not None \
           else sum(s["Pron"] for s in segs)/len(segs)
 
    return {
        "Pron": round(pron,1),
        "Acc":  round(accuracy,1),
        "Flu":  round(fluency,1),
        "Pro":  round(prosody,1),
        "Comp": round(comp,1) if comp is not None else None,
        "Words": words
    }

def print_summary(res: Dict[str, Any]) -> None:
    ins = [w["Word"] for w in res["Words"] if w["ErrorType"]=="Insertion"]
    om  = [w["Word"] for w in res["Words"] if w["ErrorType"]=="Omission"]
    mis = [w["Word"] for w in res["Words"]
           if w["ErrorType"]=="None" and w["Accuracy"]<70]
 
    scores = f"Pron {res['Pron']} | Acc {res['Acc']} | Flu {res['Flu']} | Pro {res['Pro']}"
    if res["Comp"] is not None:
        scores += f" | Comp {res['Comp']}"
 
    print("===== PRONUNCIATION SUMMARY =====")
    print(f"Overall scores  →  {scores}")
    print("\nWord errors:")
    print(f"  Insertions     : {len(ins)}")
    print(f"  Omissions      : {len(om)}")
    print(f"  Mispronounced (<70): {len(mis)}")
    if mis:
        preview = ", ".join(mis[:5]) + ("…" if len(mis)>5 else "")
        print(f"    → {preview}")


def calc_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    try:
        ins = [w["Word"] for w in res["Words"] if w["ErrorType"]=="Insertion"]
        om  = [w["Word"] for w in res["Words"] if w["ErrorType"]=="Omission"]
        mis = [w["Word"] for w in res["Words"]
            if w["ErrorType"]=="None" and w["Accuracy"]<70]
        
        scores = f"Pron {res['Pron']} | Acc {res['Acc']} | Flu {res['Flu']} | Pro {res['Pro']}"
        if res["Comp"] is not None:
            scores += f" | Comp {res['Comp']}"

        metric_data = {
            'insertions': len(ins),
            'omissions': len(om),
            'mispronounced': len(mis),
            'pronunciation': res['Pron'],
            'accuracy': res['Acc'],
            'fluency': res['Flu'],
            'prosody': res['Pro'],
            'comprehension': res['Comp'] if res['Comp'] is not None else "None Detected"
        }
        return metric_data
    except Exception as e:
        print(f"calc_metrics error: {e}")
        return None


# Acutal method to get metrics

def get_metrics(wav_path: str) -> Dict[str, Any]:
    wav = ensure_wav(wav_path)
    validate_wav(wav)
    raw = run_assessment(wav)
    if not raw:
        return False
 
    segments = [parse_segment(js) for js in raw]
    summary  = aggregate(segments)
    metric_data = calc_metrics(summary)
    return metric_data


