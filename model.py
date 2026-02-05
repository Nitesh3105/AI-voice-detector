import tempfile
import base64
import io
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


class AudioFeatureExtractor:
    """Extract audio features for voice detection"""
    
    def __init__(self, sr=22050, n_mfcc=20):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def base64_to_audio(self, audioBase64: str):
        try:
            audio_bytes = base64.b64decode(audioBase64)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=self.sr)
            return y, sr
            '''audio_buffer = io.BytesIO(audio_bytes)
            # Load audio with librosa
            y, sr = librosa.load(audio_buffer, sr=self.sr)
            return y, sr'''
        except Exception as e:
            raise ValueError(f"Error decoding audio: {str(e)}")

    def extract_features(self, audio_data, sr=None):
        """
        Extract comprehensive audio features
        
        Features extracted:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Spectral features (centroid, rolloff, bandwidth, flatness)
        - Zero crossing rate
        - Chroma features
        - Tempo and rhythm features
        - Energy and RMS
        """
        if sr is None:
            sr = self.sr
        
        features = []
        
        # MFCCs - capture timbral texture
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.n_mfcc)
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1)
        ])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        
        features.extend([
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(spectral_flatness), np.std(spectral_flatness)]
        ])
        
        # Zero crossing rate - useful for voice/unvoice detection
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.append([np.mean(zcr), np.std(zcr)])
        
        # Chroma features - harmonic content
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features.append([np.mean(rms), np.std(rms)])
        
        # Tempo (rhythm)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        features.append([tempo])
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            np.mean(mel_spec_db, axis=1),
            np.std(mel_spec_db, axis=1)
        ])
        
        # Flatten all features into a single vector
        feature_vector = np.concatenate([np.array(f).flatten() for f in features])
        
        return feature_vector

    def extract_from_base64(self, audioBase64):
        """Extract features directly from base64 string"""
        audio_data, sr = self.base64_to_audio(audioBase64)
        return self.extract_features(audio_data, sr)


class VoiceDetectionModel:
    """ML Model for detecting AI vs Human voice"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (0=human, 1=AI)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_pred, 
                                       target_names=['Human', 'AI Generated']))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_val, val_pred))
        
        return self
    
    def predict(self, X):
        """Predict labels for input features"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities for input features"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba
            return self.model.decision_function(X_scaled)
    
    def predict_from_base64(self, audioBase64):
        """
        Predict directly from base64 audio string
        
        Returns:
            dict with prediction, probability, and label
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_from_base64(audioBase64)
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.predict(features)[0]
            probabilities = self.predict_proba(features)[0]
            
            result = {
                'prediction': int(prediction),
                'label': 'AI Generated' if prediction == 1 else 'Human Voice',
                'confidence': float(probabilities[prediction]),
                'probabilities': {
                    'human': float(probabilities[0]),
                    'ai_generated': float(probabilities[1])
                }
            }
            
            return result
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None
            }
    
    def save(self, filepath):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if feature_names is not None:
                return dict(zip(feature_names, importances))
            return importances
        else:
            print("Feature importance not available for this model type")
            return None


def create_sample_dataset(n_human=100, n_ai=100, duration=3, sr=22050):
    """
    Create a synthetic dataset for demonstration
    In production, you would load real human and AI-generated audio files
    """
    print("Creating synthetic dataset for demonstration...")
    
    extractor = AudioFeatureExtractor(sr=sr)
    X = []
    y = []
    
    # Generate synthetic "human" samples
    # Real human voice has more irregularities and natural variation
    for i in range(n_human):
        # Simulate human voice with varying harmonics and noise
        t = np.linspace(0, duration, int(sr * duration))
        
        # Base frequency with slight variations (simulating human pitch variation)
        base_freq = 150 + np.random.randn() * 20
        signal = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics (typical in human voice)
        for harmonic in [2, 3, 4]:
            signal += 0.5 / harmonic * np.sin(2 * np.pi * base_freq * harmonic * t)
        
        # Add natural noise and irregularities
        signal += np.random.randn(len(signal)) * 0.1
        
        # Add amplitude modulation (breathing, emotion)
        signal *= (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
        
        features = extractor.extract_features(signal, sr)
        X.append(features)
        y.append(0)  # 0 = human
    
    # Generate synthetic "AI" samples
    # AI voice tends to be more consistent and "perfect"
    for i in range(n_ai):
        t = np.linspace(0, duration, int(sr * duration))
        
        # More consistent base frequency
        base_freq = 150
        signal = np.sin(2 * np.pi * base_freq * t)
        
        # More regular harmonics
        for harmonic in [2, 3, 4]:
            signal += 0.5 / harmonic * np.sin(2 * np.pi * base_freq * harmonic * t)
        
        # Less noise (AI voice is cleaner)
        signal += np.random.randn(len(signal)) * 0.02
        
        # More regular amplitude (less variation)
        signal *= (1 + 0.05 * np.sin(2 * np.pi * 0.5 * t))
        
        features = extractor.extract_features(signal, sr)
        X.append(features)
        y.append(1)  # 1 = AI
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("=" * 60)
    print("AI vs Human Voice Detection Model")
    print("=" * 60)
    
    # Create synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X, y = create_sample_dataset(n_human=150, n_ai=150)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Human samples: {np.sum(y == 0)}, AI samples: {np.sum(y == 1)}")
    
    # Split dataset
    print("\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\n3. Training model...")
    detector = VoiceDetectionModel(model_type='random_forest')
    detector.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_pred = detector.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, 
                               target_names=['Human', 'AI Generated']))
    
    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(f"\nTrue Negatives (Human correctly identified): {cm[0][0]}")
    print(f"False Positives (Human misclassified as AI): {cm[0][1]}")
    print(f"False Negatives (AI misclassified as Human): {cm[1][0]}")
    print(f"True Positives (AI correctly identified): {cm[1][1]}")
    
    # Save model
    print("\n5. Saving model...")
    detector.save('voice_detection_model.pkl')
    
    print("\n" + "=" * 60)
    print("Training complete! Model saved successfully.")
    print("=" * 60)
    print("\nTo use the model for prediction, see the usage example below.")
