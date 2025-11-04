import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja
SAMPLE_RATE = 16000
DURATION = 3  # sekundy
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


class EmotionDataset:
    """Klasa do ładowania i przetwarzania datasetu nEMO"""

    def __init__(self, samples_path):
        self.samples_path = samples_path
        self.label_encoder = LabelEncoder()
        self.metadata = None

    def extract_features(self, audio_path):
        """Ekstrakcja cech audio z pliku"""
        try:
            # Wczytanie audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

            # Padding lub obcięcie do stałej długości
            target_length = SAMPLE_RATE * DURATION
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            # Mel-spektrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT
            )

            # Konwersja do skali dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_spec_db

        except Exception as e:
            print(f"Błąd podczas przetwarzania {audio_path}: {e}")
            return None

    def load_dataset(self, tsv_file):
        """Ładowanie datasetu nEMO z pliku data.tsv"""
        # Wczytanie pliku TSV
        df = pd.read_csv(tsv_file, sep='\t')
        self.metadata = df

        print(f"Załadowano {len(df)} próbek")
        print(f"\nRozkład emocji:")
        print(df['emotion'].value_counts())
        print(f"\nRozkład płci:")
        print(df['gender'].value_counts())
        print(f"\nStatystyki wieku:")
        print(df['age'].describe())

        features = []
        labels = []
        valid_indices = []

        print("\n" + "=" * 50)
        print("Ładowanie i przetwarzanie plików audio...")
        print("=" * 50)

        for idx, row in df.iterrows():
            audio_path = os.path.join(self.samples_path, row['file_id'])

            if os.path.exists(audio_path):
                feat = self.extract_features(audio_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(row['emotion'])
                    valid_indices.append(idx)
            else:
                print(f"Ostrzeżenie: Nie znaleziono pliku {row['file_id']}")

            if (idx + 1) % 100 == 0:
                print(f"Przetworzono {idx + 1}/{len(df)} plików")

        print(f"\nPomyślnie przetworzono {len(features)}/{len(df)} plików")

        # Konwersja do numpy arrays
        features = np.array(features)
        features = features[..., np.newaxis]  # Dodanie wymiaru kanału

        # Enkodowanie etykiet
        labels = self.label_encoder.fit_transform(labels)

        # Zaktualizowanie metadanych tylko dla prawidłowych próbek
        self.metadata = df.iloc[valid_indices].reset_index(drop=True)

        return features, labels


class EmotionRecognitionModel:
    """Model CNN do rozpoznawania emocji"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """Budowa modelu CNN"""
        model = keras.Sequential([
            # Blok konwolucyjny 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok konwolucyjny 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok konwolucyjny 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok konwolucyjny 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Warstwa globalna
            layers.GlobalAveragePooling2D(),

            # Warstwy gęste
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Warstwa wyjściowa
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def compile_model(self):
        """Kompilacja modelu"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Trening modelu"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Trening
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Ewaluacja modelu"""
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Predykcje
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        return y_pred


def plot_history(history):
    """Wizualizacja procesu treningu"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Wizualizacja macierzy pomyłek"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


def analyze_dataset(metadata):
    """Analiza statystyczna datasetu"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Rozkład emocji
    metadata['emotion'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Rozkład emocji')
    axes[0, 0].set_xlabel('Emocja')
    axes[0, 0].set_ylabel('Liczba próbek')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Rozkład płci
    metadata['gender'].value_counts().plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Rozkład płci')
    axes[0, 1].set_xlabel('Płeć')
    axes[0, 1].set_ylabel('Liczba próbek')

    # Rozkład wieku
    axes[1, 0].hist(metadata['age'], bins=20, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Rozkład wieku')
    axes[1, 0].set_xlabel('Wiek')
    axes[1, 0].set_ylabel('Liczba próbek')

    # Emocje vs płeć
    emotion_gender = pd.crosstab(metadata['emotion'], metadata['gender'])
    emotion_gender.plot(kind='bar', ax=axes[1, 1], stacked=True)
    axes[1, 1].set_title('Emocje według płci')
    axes[1, 1].set_xlabel('Emocja')
    axes[1, 1].set_ylabel('Liczba próbek')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Płeć')

    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    plt.show()

    # Wyświetlenie statystyk tekstowych
    print("\n" + "=" * 50)
    print("STATYSTYKI DATASETU")
    print("=" * 50)
    print(f"\nŁączna liczba próbek: {len(metadata)}")
    print(f"Liczba unikalnych mówców: {metadata['speaker_id'].nunique()}")
    print(f"\nŚrednia długość tekstu: {metadata['raw_text'].str.len().mean():.1f} znaków")
    print(f"Średni wiek mówców: {metadata['age'].mean():.1f} lat")

    return fig


def predict_emotion(model, audio_path, label_encoder):
    """Predykcja emocji dla pojedynczego pliku audio"""
    dataset = EmotionDataset(None)
    features = dataset.extract_features(audio_path)

    if features is not None:
        features = features[np.newaxis, ..., np.newaxis]
        prediction = model.predict(features)
        emotion_idx = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        confidence = prediction[0][emotion_idx]

        print(f"\nPredykcja: {emotion}")
        print(f"Pewność: {confidence:.2%}")

        # Wyświetlenie wszystkich prawdopodobieństw
        print("\nWszystkie emocje:")
        for idx, prob in enumerate(prediction[0]):
            emo = label_encoder.inverse_transform([idx])[0]
            print(f"{emo}: {prob:.2%}")

        return emotion, confidence

    return None, None


# Główna funkcja
def main():
    # Ścieżki do danych nEMO
    SAMPLES_PATH = "samples"  # Folder z plikami .wav
    TSV_FILE = "data.tsv"  # Plik z metadanymi

    # 1. Ładowanie datasetu
    print("=" * 50)
    print("ŁADOWANIE DATASETU nEMO")
    print("=" * 50)

    dataset = EmotionDataset(SAMPLES_PATH)
    X, y = dataset.load_dataset(TSV_FILE)

    print(f"\nKształt features: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print(f"Klasy emocji: {dataset.label_encoder.classes_}")

    # Analiza datasetu
    #analyze_dataset(dataset.metadata)

    # 2. Podział na zbiory
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nTrain set: {X_train.shape[0]}")
    print(f"Validation set: {X_val.shape[0]}")
    print(f"Test set: {X_test.shape[0]}")

    # 3. Budowa i trening modelu
    print("\n" + "=" * 50)
    print("BUDOWA I TRENING MODELU")
    print("=" * 50)

    model = EmotionRecognitionModel(
        input_shape=X_train.shape[1:],
        num_classes=len(np.unique(y))
    )

    model.compile_model()
    print("\nArchitektura modelu:")
    model.model.summary()

    # Trening
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # 4. Ewaluacja
    print("\n" + "=" * 50)
    print("EWALUACJA MODELU")
    print("=" * 50)

    y_pred = model.evaluate(X_test, y_test)

    # Raport klasyfikacji
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=dataset.label_encoder.classes_
    ))

    # Wizualizacje
    plot_history(history)
    plot_confusion_matrix(y_test, y_pred, dataset.label_encoder.classes_)

    # 5. Zapisanie modelu
    model.model.save('emotion_recognition_model.h5')
    print("\nModel zapisany jako 'emotion_recognition_model.h5'")

    # Zapisanie label encodera
    np.save('label_encoder_classes.npy', dataset.label_encoder.classes_)
    print("Label encoder zapisany jako 'label_encoder_classes.npy'")

    return model, dataset.label_encoder


if __name__ == "__main__":
    # Uruchomienie treningu
    model, label_encoder = main()

    # Przykład predykcji dla nowego pliku
    # predict_emotion(model.model, "path/to/test_audio.wav", label_encoder)