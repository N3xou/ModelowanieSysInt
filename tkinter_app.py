import os
import torch
import librosa
import numpy as np
import sounddevice as sd
from torch import nn
from train import Model, pad_or_trunc
import time
#import gradio as gr
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
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

recording = False
mic_buffer = []

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
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, T]
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

def predict_emotion(audio, pad = 0):
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-4:
        return "No sound detected"
    if pad == 1:
        audio = pad_or_trunc(audio)
    mel = extract_mel(audio)
    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        emotion = classes[idx.item()]
        confidence = conf.item()*100
    return f"{emotion} \n Confidence: {confidence:.2f}%"


def predict_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return
    try:
        audio, _ = librosa.load(file_path, sr=SR)

        if len(audio) > MAX_SAMPLES:
            emotion, avg_conf, dist = predict_long_audio(audio)

            dist_text = "\n".join([f"{k}: {v}" for k, v in dist.items()])

            result_label.config(
                text=f"Prediction: {emotion}\nAvg confidence: {avg_conf:.2f}%\n\nDistribution:\n{dist_text}"
            )
        else:
            result = predict_emotion(audio, 1)
            result_label.config(text=f"Prediction: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def predict_from_microphone():
    global recording, mic_buffer

    if not recording:
        # START recording
        recording = True
        mic_buffer = []

        status_label.config(text="Recording... Press again to stop", fg="green")
        status_label.update_idletasks()

        def record_loop():
            while recording:
                chunk = sd.rec(int(0.5 * SR), samplerate=SR, channels=1, dtype='float32')
                sd.wait()
                mic_buffer.append(chunk.flatten())

        threading.Thread(target=record_loop, daemon=True).start()

    else:
        # STOP recording
        recording = False
        status_label.config(text="Processing...", fg="blue")
        status_label.update_idletasks()

        audio = np.concatenate(mic_buffer) if mic_buffer else np.array([])

        if len(audio) < MAX_SAMPLES:
            messagebox.showwarning("Too short", "Recording must be at least 4 seconds long.")
            status_label.config(text="")
            return

        try:
            emotion, avg_conf, dist = predict_long_audio(audio)

            dist_text = "\n".join([f"{k}: {v}" for k, v in dist.items()])

            result_label.config(
                text=f"Prediction: {emotion}\nAvg confidence: {avg_conf:.2f}%\n\nDistribution:\n{dist_text}"
            )
            status_label.config(text="")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            status_label.config(text="")


from collections import Counter, defaultdict

def predict_long_audio(audio):
    chunk_size = MAX_SAMPLES  # 4 seconds
    total_len = len(audio)

    num_chunks = total_len // chunk_size  # discard remainder

    if num_chunks == 0:
        return predict_emotion(audio, pad=1)

    emotions = []
    confidences = defaultdict(list)

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]

        mel = extract_mel(chunk)

        with torch.no_grad():
            logits = model(mel)
            probs = torch.softmax(logits, dim=1)

            conf, idx = torch.max(probs, dim=1)
            emotion = classes[idx.item()]
            confidence = conf.item() * 100

        emotions.append(emotion)
        confidences[emotion].append(confidence)

    # Most common emotion
    final_emotion = Counter(emotions).most_common(1)[0][0]

    # Average confidence of that emotion
    avg_confidence = np.mean(confidences[final_emotion])

    # Optional: emotion distribution
    emotion_percentages = {
        emo: f"{(count/num_chunks)*100:.1f}%"
        for emo, count in Counter(emotions).items()
    }

    return final_emotion, avg_confidence, emotion_percentages

# -----------------------
# Tkinter GUI
# -----------------------
def center_window(win, width=600, height=250):
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    win.geometry(f"{width}x{height}+{x}+{y}")

root = tk.Tk()
root.title("🎤 Speech Emotion Recognition")
center_window(root, 600,250)
root.configure(bg="#f0f0f0")

# Header
header = tk.Label(root, text="Speech Emotion Recognition", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
header.pack(pady=10)

# Buttons frame
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=10)

btn_file = tk.Button(frame_buttons, text="📂 Predict from WAV file", command=predict_from_file,
                     width=25, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
btn_file.grid(row=0, column=0, padx=10, pady=5)

btn_mic = tk.Button(
    frame_buttons,
    text="🎙️ Start Recording",
    command=predict_from_microphone,
    width=25,
    height=2,
    bg="#2196F3",
    fg="white",
    font=("Helvetica", 12, "bold")
)
btn_mic.grid(row=0, column=1, padx=10, pady=5)

btn_mic.grid(row=0, column=1, padx=10, pady=5)

# Result label
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14), bg="#f0f0f0")
result_label.pack(pady=15)

# Status label
status_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f0f0")
status_label.pack(pady=5)

root.mainloop()