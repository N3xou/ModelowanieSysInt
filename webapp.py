import os
import io
import base64
import torch
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
from flask import Flask, render_template, request, jsonify
from torch import nn
from train import Model, pad_or_trunc
from train_od_jakuba import ResNetInspired

# -----------------------
# Constants
# -----------------------
SR = 24000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_SECONDS = 4
MAX_SAMPLES = int(SR * MAX_SECONDS)

MODEL_PATH = "best-0.76.pt"

app = Flask(__name__)


# -----------------------
# Utilities
# -----------------------

def extract_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel


# -----------------------
# Load model
# -----------------------
def load_model(model_path=MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["labels"]
    model = ResNetInspired(num_classes=len(classes))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, classes


model, classes = load_model()


def predict_emotion(audio, pad=0):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-4:
        return {"emotion": "No sound detected", "confidence": 0}

    if pad == 1:
        audio = pad_or_trunc(audio)

    mel = extract_mel(audio)
    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        emotion = classes[idx.item()]
        confidence = conf.item()*100

    return {"emotion": emotion, "confidence": f"{confidence:.2f}"}


# -----------------------
# Routes
# -----------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read audio file
        audio_bytes = file.read()
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        if samplerate != SR:
            audio = librosa.resample(audio, orig_sr=samplerate, target_sr=SR)

        result = predict_emotion(audio, 1)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_microphone', methods=['POST'])
def predict_microphone():
    try:
        data = request.json
        audio_data = data.get('audio')

        if not audio_data:
            return jsonify({"error": "No audio data"}), 400

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR)

        result = predict_emotion(audio, 0)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)