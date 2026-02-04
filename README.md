# AI-voice-detector
API model detects mp3 base64 audio whether ai generated or human voice
Multilingual
● Tamil
● English
● Hindi
● Malayalam
● Telugu
# AI vs Human Voice Detection Model

A machine learning model to detect whether an audio sample contains human voice or AI-generated voice. The model accepts base64-encoded MP3/WAV audio files and returns predictions with confidence scores.

## Features

- **Audio Feature Extraction**: Extracts 200+ audio features including MFCCs, spectral features, chroma, tempo, and more
- **Multiple ML Models**: Supports Random Forest, Gradient Boosting, and SVM classifiers
- **Base64 Support**: Direct prediction from base64-encoded audio strings
- **REST API**: Flask API for easy integration
- **Batch Processing**: Support for processing multiple audio files at once

## Architecture

The model works in three main steps:

1. **Audio Preprocessing**: Converts base64 string to audio waveform
2. **Feature Extraction**: Extracts comprehensive audio features
   - MFCCs (Mel-frequency cepstral coefficients)
   - Spectral features (centroid, rolloff, bandwidth, flatness)
   - Zero crossing rate
   - Chroma features
   - Tempo and rhythm
   - Energy and RMS
   - Mel spectrogram statistics
3. **Classification**: Predicts whether voice is human or AI-generated

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with break-system-packages flag if needed
pip install -r requirements.txt --break-system-packages
```

## Quick Start

### 1. Train the Model

```bash
python voice_detector.py
```

This will:
- Generate a synthetic training dataset
- Train a Random Forest classifier
- Evaluate the model
- Save the trained model to `voice_detection_model.pkl`

### 2. Test the Model

```bash
python usage_example.py
```

This demonstrates:
- Loading the trained model
- Creating test audio samples
- Making predictions from audio files
- Making predictions from base64 strings

### 3. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

## Usage Examples

### Python Usage

```python
from voice_detector import VoiceDetectionModel
import base64

# Load model
detector = VoiceDetectionModel()
detector.load('voice_detection_model.pkl')

# Convert audio file to base64
with open('audio.mp3', 'rb') as f:
    audio_bytes = f.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

# Make prediction
result = detector.predict_from_base64(base64_audio)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Human probability: {result['probabilities']['human']:.2%}")
print(f"AI probability: {result['probabilities']['ai_generated']:.2%}")
```

### API Usage

#### Using cURL

```bash
# Single prediction
curl -X POST http://localhost:5000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"audio": "BASE64_ENCODED_AUDIO_HERE"}'
```



## API Endpoints

### GET /
Returns API documentation and usage examples

### GET /health
Health check endpoint
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### POST /api/v1/detect
Detect single audio sample

**Request:**
```json
{
    "audio": "base64_encoded_audio_string"
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 0,
    "label": "Human Voice",
    "confidence": 0.95,
    "probabilities": {
        "human": 0.95,
        "ai_generated": 0.05
    }
}
```

### POST /api/v1/batch-detect
Detect multiple audio samples

**Request:**
```json
{
    "audios": [
        "base64_audio_1",
        "base64_audio_2"
    ]
}
```

**Response:**
```json
{
    "success": true,
    "count": 2,
    "results": [
        {
            "index": 0,
            "prediction": 0,
            "label": "Human Voice",
            "confidence": 0.95,
            "probabilities": {
                "human": 0.95,
                "ai_generated": 0.05
            }
        },
        ...
    ]
}
```

### GET /api/v1/model-info
Get model information

**Response:**
```json
{
    "success": true,
    "model_type": "random_forest",
    "feature_extractor": {
        "sample_rate": 22050,
        "n_mfcc": 20
    }
}
```

## Model Performance

The synthetic dataset shows example performance. With real data, you should:

1. **Collect Real Data**: Gather actual human voice recordings and AI-generated voice samples
2. **Balance Dataset**: Ensure equal representation of both classes
3. **Feature Engineering**: Tune feature extraction parameters for your specific use case
4. **Model Tuning**: Experiment with different models and hyperparameters

Expected metrics with real data:
- Accuracy: 85-95% (depending on dataset quality)
- Precision: 85-95%
- Recall: 85-95%

## Training with Your Own Data

To train with real audio files:

```python
from voice_detector import VoiceDetectionModel, AudioFeatureExtractor
import numpy as np
import os

# Initialize feature extractor
extractor = AudioFeatureExtractor()

# Extract features from your audio files
X = []
y = []

# Human voice samples
for audio_file in os.listdir('human_voices/'):
    audio_path = f'human_voices/{audio_file}'
    with open(audio_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    features = extractor.extract_from_base64(audio_b64)
    X.append(features)
    y.append(0)  # 0 = human

# AI-generated voice samples
for audio_file in os.listdir('ai_voices/'):
    audio_path = f'ai_voices/{audio_file}'
    with open(audio_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    features = extractor.extract_from_base64(audio_b64)
    X.append(features)
    y.append(1)  # 1 = AI

X = np.array(X)
y = np.array(y)

# Split and train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
detector = VoiceDetectionModel(model_type='random_forest')
detector.train(X_train, y_train, X_test, y_test)

# Save model
detector.save('custom_voice_model.pkl')
```

## Key Distinguishing Features

The model looks for several key differences between human and AI voices:

### Human Voice Characteristics:
- Natural pitch variations
- Breathing patterns
- Emotional fluctuations
- Slight irregularities in timing
- More complex harmonic structure
- Background noise and artifacts

### AI Voice Characteristics:
- More consistent pitch
- Regular timing
- Cleaner audio (less noise)
- More uniform amplitude
- Sometimes overly perfect harmonics
- Lack of natural breathing patterns

## Limitations

1. **Quality Dependent**: Performance depends on audio quality and recording conditions
2. **AI Evolution**: Newer AI voice models are increasingly realistic
3. **Dataset Bias**: Model trained on specific types of voices may not generalize well
4. **Synthetic Training Data**: The included example uses synthetic data - real data is needed for production

## Improving Accuracy

1. **More Training Data**: Use larger, diverse datasets
2. **Advanced Features**: Add deep learning features (spectrograms with CNNs)
3. **Ensemble Methods**: Combine multiple models
4. **Regular Updates**: Retrain as AI voice technology evolves
5. **Domain-Specific Training**: Train on specific voice types (news, podcasts, etc.)

## Contributing

To improve this model:
1. Provide real human and AI voice datasets
2. Experiment with different features
3. Try deep learning approaches (CNN, RNN, Transformer)
4. Add support for more audio formats

## License

MIT License - Feel free to use and modify for your needs

## Citation

If you use this model in your research or project, please cite appropriately.

## Support

For issues or questions:
- Check the examples in `usage_example.py`
- Review the API documentation at `http://localhost:5000/`
- Submit issues with detailed audio examples

## Future Enhancements

- [ ] Deep learning models (CNN, Transformer)
- [ ] Real-time audio stream processing
- [ ] More audio format support
- [ ] Explainability features (which audio characteristics led to the prediction)
- [ ] Pre-trained models on large datasets
- [ ] Multi-language support
- [ ] Voice cloning detection
