import os
import pickle
import librosa
import numpy as np
import sounddevice as sd
from keras.models import model_from_json
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from collections import Counter, defaultdict

# -----------------------
# Constants
# -----------------------
SR = 22050  # Sample rate zgodny z treningiem
DURATION = 2.5  # Długość fragmentu audio (sekundy)
OFFSET = 0.6  # Offset początkowy (sekundy)
FRAME_LENGTH = 2048
HOP_LENGTH = 512
INPUT_LENGTH = 2376  # Zgodnie z architekturą modelu

MODEL_JSON_PATH = "CNN_model.json"
MODEL_WEIGHTS_PATH = "best_model1_weights.h5"
SCALER_PATH = "scaler2.pickle"
ENCODER_PATH = "encoder2.pickle"

recording = False
mic_buffer = []


# -----------------------
# Load model and preprocessors
# -----------------------
def load_model():
    """Ładuje model Keras wraz ze scalerem i encoderem"""
    # Wczytaj architekturę modelu
    with open(MODEL_JSON_PATH, 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    # Wczytaj wagi
    model.load_weights(MODEL_WEIGHTS_PATH)

    # Wczytaj scaler i encoder
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)

    print(f"✓ Model załadowany pomyślnie!")
    print(f"✓ Klasy: {list(encoder.categories_[0])}")

    return model, scaler, encoder


# Załaduj model globalnie
model, scaler, encoder = load_model()
classes = list(encoder.categories_[0])


# -----------------------
# Feature extraction - DOKŁADNIE JAK W ORYGINALNYM NOTEBOOKU
# -----------------------
def zcr(data, frame_length, hop_length):
    """Zero Crossing Rate"""
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    """Root Mean Square Energy"""
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    """Mel-frequency cepstral coefficients"""
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    """
    Ekstrahuje cechy audio DOKŁADNIE JAK W ORYGINALNYM NOTEBOOKU.
    Zwraca: ZCR + RMSE + MFCC (spłaszczone)
    """
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def get_predict_feat(audio_data, sr=SR):
    """
    Preprocessing audio do predykcji - DOKŁADNIE JAK W ORYGINALNYM NOTEBOOKU.

    Args:
        audio_data: numpy array z audio
        sr: sample rate

    Returns:
        numpy array w kształcie (1, 2376, 1) gotowy do predykcji
    """
    # Ekstrahuj cechy
    res = extract_features(audio_data, sr)

    # Reshape do (1, 2376)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, INPUT_LENGTH))

    # Skaluj
    i_result = scaler.transform(result)

    # Expand dims do (1, 2376, 1) dla CNN
    final_result = np.expand_dims(i_result, axis=2)

    return final_result


# -----------------------
# Prediction functions
# -----------------------
def predict_emotion(audio, sr=SR, use_offset=True):
    """
    Predykcja emocji z fragmentu audio.

    Args:
        audio: numpy array z audio
        sr: sample rate
        use_offset: czy stosować offset i duration jak w treningu

    Returns:
        (emotion_name, confidence)
    """
    # Sprawdź czy audio nie jest puste
    rms_val = np.sqrt(np.mean(audio ** 2))
    if rms_val < 1e-4:
        return "No sound detected", 0.0

    # Jeśli use_offset=True, wytnij fragment zgodnie z treningiem
    if use_offset and len(audio) > int((OFFSET + DURATION) * sr):
        start_sample = int(OFFSET * sr)
        end_sample = int((OFFSET + DURATION) * sr)
        audio = audio[start_sample:end_sample]

    # Preprocessing
    try:
        features = get_predict_feat(audio, sr)
    except Exception as e:
        print(f"Błąd podczas preprocessingu: {e}")
        return "Error in preprocessing", 0.0

    # Sprawdź kształt
    if features.shape != (1, INPUT_LENGTH, 1):
        print(f"OSTRZEŻENIE: Niepoprawny kształt features: {features.shape}")
        return "Error: Invalid shape", 0.0

    # Predykcja
    predictions = model.predict(features, verbose=0)

    # Dekoduj wynik
    y_pred = encoder.inverse_transform(predictions)
    emotion = y_pred[0][0]

    # Pobierz confidence (prawdopodobieństwo najlepszej klasy)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx] * 100

    return emotion, confidence


def predict_long_audio(audio, sr=SR):
    """
    Predykcja dla długiego audio poprzez podział na fragmenty.
    Każdy fragment to 2.5s z offset=0.6s (zgodnie z treningiem).
    """
    # Oblicz ile fragmentów możemy wyciąć
    fragment_duration = DURATION  # 2.5s
    step_duration = DURATION  # Przesunięcie między fragmentami (bez nakładania)

    fragment_samples = int(fragment_duration * sr)
    step_samples = int(step_duration * sr)

    total_samples = len(audio)

    # Pomiń pierwsze OFFSET sekund
    start_offset = int(OFFSET * sr)
    audio = audio[start_offset:]
    total_samples = len(audio)

    # Liczba fragmentów
    num_fragments = (total_samples - fragment_samples) // step_samples + 1

    if num_fragments < 1:
        # Za krótkie audio - spróbuj predykcji na tym co jest
        emotion, conf = predict_emotion(audio, sr, use_offset=False)
        return emotion, conf, {emotion: "100.0%"}

    emotions = []
    confidences = defaultdict(list)

    for i in range(num_fragments):
        start = i * step_samples
        end = start + fragment_samples

        if end > total_samples:
            break

        chunk = audio[start:end]

        emotion, confidence = predict_emotion(chunk, sr, use_offset=False)

        if emotion != "No sound detected" and emotion != "Error in preprocessing":
            emotions.append(emotion)
            confidences[emotion].append(confidence)

    if not emotions:
        return "No sound detected", 0.0, {}

    # Najczęstsza emocja
    final_emotion = Counter(emotions).most_common(1)[0][0]

    # Średnia pewność dla tej emocji
    avg_confidence = np.mean(confidences[final_emotion])

    # Rozkład procentowy emocji
    emotion_percentages = {
        emo: f"{(count / len(emotions)) * 100:.1f}%"
        for emo, count in Counter(emotions).items()
    }

    return final_emotion, avg_confidence, emotion_percentages


# -----------------------
# GUI Functions
# -----------------------
def predict_from_file():
    """Predykcja z pliku audio"""
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.flac")]
    )
    if not file_path:
        return

    try:
        # Wczytaj audio z oryginalnym SR
        audio, sr_loaded = librosa.load(file_path, sr=SR)

        # Oblicz całkowitą długość
        total_duration = len(audio) / sr_loaded

        if total_duration <= (OFFSET + DURATION):
            # Krótkie audio - pojedyncza predykcja
            emotion, confidence = predict_emotion(audio, sr_loaded, use_offset=True)

            result_label.config(
                text=f"Prediction: {emotion}\nConfidence: {confidence:.2f}%"
            )
        else:
            # Długie audio - podziel na fragmenty
            emotion, avg_conf, dist = predict_long_audio(audio, sr_loaded)

            dist_text = "\n".join([f"{k}: {v}" for k, v in dist.items()])

            result_label.config(
                text=f"Prediction: {emotion}\nAvg confidence: {avg_conf:.2f}%\n\nDistribution:\n{dist_text}"
            )

    except Exception as e:
        messagebox.showerror("Error", f"Error processing file:\n{str(e)}")
        import traceback
        traceback.print_exc()


def predict_from_microphone():
    """Nagrywanie i predykcja z mikrofonu"""
    global recording, mic_buffer

    if not recording:
        # START nagrywania
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
        # STOP nagrywania
        recording = False
        status_label.config(text="Processing...", fg="blue")
        status_label.update_idletasks()

        audio = np.concatenate(mic_buffer) if mic_buffer else np.array([])

        min_duration = OFFSET + DURATION  # 3.1s
        if len(audio) < SR * min_duration:
            messagebox.showwarning(
                "Too short",
                f"Recording must be at least {min_duration:.1f} seconds long."
            )
            status_label.config(text="")
            return

        try:
            emotion, avg_conf, dist = predict_long_audio(audio, SR)

            dist_text = "\n".join([f"{k}: {v}" for k, v in dist.items()])

            result_label.config(
                text=f"Prediction: {emotion}\nAvg confidence: {avg_conf:.2f}%\n\nDistribution:\n{dist_text}"
            )
            status_label.config(text="")

        except Exception as e:
            messagebox.showerror("Error", f"Error processing audio:\n{str(e)}")
            status_label.config(text="")
            import traceback
            traceback.print_exc()


# -----------------------
# Tkinter GUI
# -----------------------
def center_window(win, width=600, height=320):
    """Centruje okno na ekranie"""
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    win.geometry(f"{width}x{height}+{x}+{y}")


# Główne okno
root = tk.Tk()
root.title("🎤 Speech Emotion Recognition (Keras - FIXED)")
center_window(root, 600, 320)
root.configure(bg="#f0f0f0")

# Header
header = tk.Label(
    root,
    text="Speech Emotion Recognition",
    font=("Helvetica", 18, "bold"),
    bg="#f0f0f0"
)
header.pack(pady=10)

# Subtitle
subtitle = tk.Label(
    root,
    text="Using Keras CNN Model (CORRECTED PREPROCESSING)",
    font=("Helvetica", 10, "italic"),
    bg="#f0f0f0",
    fg="#666"
)
subtitle.pack()

# Info about preprocessing
info_preproc = tk.Label(
    root,
    text=f"Features: ZCR + RMSE + MFCC | Duration: {DURATION}s | Offset: {OFFSET}s",
    font=("Helvetica", 9),
    bg="#f0f0f0",
    fg="#999"
)
info_preproc.pack(pady=5)

# Buttons frame
frame_buttons = tk.Frame(root, bg="#f0f0f0")
frame_buttons.pack(pady=15)

btn_file = tk.Button(
    frame_buttons,
    text="📂 Predict from Audio File",
    command=predict_from_file,
    width=25,
    height=2,
    bg="#4CAF50",
    fg="white",
    font=("Helvetica", 12, "bold")
)
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

# Result label
result_label = tk.Label(
    root,
    text="Prediction: ",
    font=("Helvetica", 14),
    bg="#f0f0f0",
    justify="left"
)
result_label.pack(pady=15)

# Status label
status_label = tk.Label(
    root,
    text="",
    font=("Helvetica", 12),
    bg="#f0f0f0"
)
status_label.pack(pady=5)

# Info footer
info_label = tk.Label(
    root,
    text=f"Model: CNN | Classes: {len(classes)} | SR: {SR}Hz",
    font=("Helvetica", 9),
    bg="#f0f0f0",
    fg="#999"
)
info_label.pack(side="bottom", pady=5)

root.mainloop()