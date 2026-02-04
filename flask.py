from fastapi import FastAPI, Depends
from pydantic import BaseModel
import model

app = FastAPI(title="AI Voice Detection API")

import os

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not set")
from fastapi import Header
from fastapi.responses import JSONResponse

class VoiceRequest(BaseModel):
    audioBase64: str
    audioFormat: str = "mp3"
    language: str

@app.post("/v1/voice/analyze")
def verify_api_key(
    payload: VoiceRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Invalid API key"
            }
        )


def analyze_voice(
    payload: VoiceRequest,
    _: str = Depends(verify_api_key)
):
    y, sr = decode_base64_mp3(payload.audioBase64)

    model = VoiceDetectionModel()
	model.load("voice_detection_model.pkl")
	result=model.predict_from_base64(audioBase64)

    return {
    "status": "success",
    "classification": result['label'],
    "confidence": round(result['confidence']),
    "explanation": (f" After considering- MFCCs (Mel-frequency cepstral coefficients),- Spectral features (centroid, rolloff, bandwidth, flatness),- Zero crossing rate,- Chroma features,- Tempo and rhythm features,- Energy and RMS it seems that it is {result['label']} voice"),
    "language": payload.language}
