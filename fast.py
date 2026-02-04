
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import model

app = FastAPI(title="AI Voice Detection API")

# Get API key from environment
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not set")


class VoiceRequest(BaseModel):
    audioBase64: str
    audioFormat: str = "mp3"
    language: str


def verify_api_key(x_api_key: str = Header(...)):
    """Verify the API key from request headers"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


@app.post("/v1/voice/analyze")
def analyze_voice(
    payload: VoiceRequest,
    x_api_key: str = Header(...)
):
    """Analyze voice audio to detect if it's AI-generated or human"""
    
    # Verify API key
    if x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Invalid API key"
            }
        )
    
    try:
        # Load the model
        voice_model = model.VoiceDetectionModel()
        voice_model.load("voice_detection_model.pkl")
        
        # Predict from base64 audio
        result = voice_model.predict_from_base64(payload.audioBase64)
        
        # Check for errors in prediction
        if 'error' in result:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Error processing audio: {result['error']}"
                }
            )
        
        # Return successful response
        return {
            "status": "success",
            "classification": result['label'],
            "confidence": round(result['confidence'] * 100, 2),  # Convert to percentage
            "explanation": (
                f"After analyzing MFCCs (Mel-frequency cepstral coefficients), "
                f"Spectral features (centroid, rolloff, bandwidth, flatness), "
                f"Zero crossing rate, Chroma features, Tempo and rhythm features, "
                f"and Energy/RMS, the audio appears to be {result['label']}."
            ),
            "language": payload.language,
            "probabilities": {
                "human": round(result['probabilities']['human'] * 100, 2),
                "ai_generated": round(result['probabilities']['ai_generated'] * 100, 2)
            }
        }
    
    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Model file not found. Please train the model first."
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Voice Detection API"}
