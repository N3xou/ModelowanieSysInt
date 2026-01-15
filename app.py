import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from NN_models import EmotionModels

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Konfiguracja GPU
print("=" * 70)
print("KONFIGURACJA GPU")
print("=" * 70)

# Ustawienie zmiennych środowiskowych dla CUDA (jeśli potrzebne)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ukrycie ostrzeżeń TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Wyłączenie oneDNN

# Wymuś użycie GPU jeśli dostępne
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Sprawdzenie dostępności GPU
print(f"Wersja TensorFlow: {tf.__version__}")
print(f"Zbudowany z CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Włączenie dynamicznego przydzielania pamięci GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"\n✅ Znaleziono {len(gpus)} GPU:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")

        # Ustawienie GPU jako domyślnego urządzenia
        logic   al_gpus = tf.config.list_logical_devices('GPU')
        print(f"   Logical GPUs: {len(logical_gpus)}")

        # Wyświetlenie szczegółów GPU
        try:
            from tensorflow.python.platform import build_info

            print(f"\n📊 Build info:")
            print(f"   CUDA version: {build_info.build_info.get('cuda_version', 'N/A')}")
            print(f"   cuDNN version: {build_info.build_info.get('cudnn_version', 'N/A')}")
        except:
            pass

    except RuntimeError as e:
        print(f"❌ Błąd konfiguracji GPU: {e}")
        print("⚠️  Trening będzie wykonany na CPU")
else:
    print("\n⚠️  Nie znaleziono GPU - używam CPU")
    print("\n📝 Rozwiązanie problemu:")
    print("   1. Dodaj CUDA do PATH:")
    print("      - Win+R → sysdm.cpl → Zaawansowane → Zmienne środowiskowe")
    print("      - Dodaj: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin")
    print("   2. Skopiuj pliki cuDNN do folderu CUDA (bin, include, lib)")
    print("   3. Zainstaluj pakiety:")
    print("      pip install nvidia-cudnn-cu12==9.1.0.70")
    print("      pip install nvidia-cuda-runtime-cu12==12.4.127")
    print("   4. ZRESTARTUJ PowerShell/terminal")

# Test GPU
print("\n🧪 Test GPU:")
try:
    # Sprawdzenie czy GPU jest dostępne do obliczeń
    if gpus:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("✅ GPU działa poprawnie dla obliczeń TensorFlow!")
            print(f"   Device: {c.device}")
    else:
        # Test na CPU
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("ℹ️  Obliczenia wykonywane na CPU")
        print(f"   Device: {c.device}")
except Exception as e:
    print(f"❌ Test nie powiódł się: {e}")

print("=" * 70 + "\n")

# Konfiguracja
SAMPLE_RATE = 24000 # liczba probek na sekunde
DURATION = 6 # sekundy
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
        """Ekstrakcja cech audio z pliku - ulepszona wersja"""
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

            # Padding lub obcięcie
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

            # Konwersja do skali dB i normalizacja
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalizacja do zakresu [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

            return mel_spec_norm

        except Exception as e:
            print(f"Błąd podczas przetwarzania {audio_path}: {e}")
            return None

    def load_dataset(self, tsv_file):
        """Ładowanie datasetu nEMO z pliku data.tsv"""
        df = pd.read_csv(tsv_file, sep='\t')
        self.metadata = df

        print(f"Załadowano {len(df)} próbek")
        print(f"\nRozkład emocji:")
        emotion_counts = df['emotion'].value_counts()
        print(emotion_counts)

        # Sprawdzenie balansu klas
        print(f"\nRozkład procentowy:")
        print((emotion_counts / len(df) * 100).round(2))

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

        features = np.array(features)
        features = features[..., np.newaxis]

        # Enkodowanie etykiet
        labels = self.label_encoder.fit_transform(labels)

        self.metadata = df.iloc[valid_indices].reset_index(drop=True)

        return features, labels



def plot_class_distribution(y, label_encoder, title="Rozkład klas"):
    """Wizualizacja rozkładu klas"""
    class_counts = Counter(y)
    classes = label_encoder.classes_
    counts = [class_counts[i] for i in range(len(classes))]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')

    # Dodanie wartości na słupkach
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}\n({height / sum(counts) * 100:.1f}%)',
                 ha='center', va='bottom')

    plt.xlabel('Emocja', fontsize=12)
    plt.ylabel('Liczba próbek', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_fold_results(fold_histories, fold_scores):
    """Wizualizacja wyników z wszystkich foldów"""
    n_folds = len(fold_histories)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy dla każdego foldu
    for i, history in enumerate(fold_histories):
        axes[0, 0].plot(history.history['accuracy'], label=f'Fold {i + 1} Train', alpha=0.6)
        axes[0, 0].plot(history.history['val_accuracy'], label=f'Fold {i + 1} Val', alpha=0.6, linestyle='--')
    axes[0, 0].set_title('Accuracy - wszystkie foldy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # Loss dla każdego foldu
    for i, history in enumerate(fold_histories):
        axes[0, 1].plot(history.history['loss'], label=f'Fold {i + 1} Train', alpha=0.6)
        axes[0, 1].plot(history.history['val_loss'], label=f'Fold {i + 1} Val', alpha=0.6, linestyle='--')
    axes[0, 1].set_title('Loss - wszystkie foldy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    # Porównanie accuracy między foldami
    fold_nums = [f'Fold {i + 1}' for i in range(n_folds)]
    train_accs = [score['train_accuracy'] for score in fold_scores]
    val_accs = [score['val_accuracy'] for score in fold_scores]

    x = np.arange(len(fold_nums))
    width = 0.35
    axes[1, 0].bar(x - width / 2, train_accs, width, label='Train', color='skyblue')
    axes[1, 0].bar(x + width / 2, val_accs, width, label='Validation', color='lightcoral')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Porównanie accuracy między foldami')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(fold_nums)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Średnie wyniki
    avg_train = np.mean(train_accs)
    avg_val = np.mean(val_accs)
    std_train = np.std(train_accs)
    std_val = np.std(val_accs)

    axes[1, 1].bar(['Train', 'Validation'], [avg_train, avg_val],
                   yerr=[std_train, std_val], capsize=10,
                   color=['skyblue', 'lightcoral'], edgecolor='black')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Średnia accuracy ze wszystkich foldów')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Dodanie wartości na słupkach
    for i, (v, err) in enumerate(zip([avg_train, avg_val], [std_train, std_val])):
        axes[1, 1].text(i, v + err + 0.02, f'{v:.3f}\n±{err:.3f}',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('kfold_results.png', dpi=300, bbox_inches='tight')
    #plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, fold_num=None):
    """Wizualizacja macierzy pomyłek"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Macierz surowa
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    title1 = 'Confusion Matrix (liczby)'
    if fold_num is not None:
        title1 += f' - Fold {fold_num}'
    ax1.set_title(title1)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Macierz znormalizowana
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    title2 = 'Confusion Matrix (normalized)'
    if fold_num is not None:
        title2 += f' - Fold {fold_num}'
    ax2.set_title(title2)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    if fold_num is not None:
        plt.savefig(f'confusion_matrix_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    #plt.show()

### USTAWIENIA TRENOWANIA, NIE ZAPOMNIJ O TYPU MODELI NA DOLE
TRAIN_LEARNRATE=0.00001
TRAIN_EPOCHS=512
TRAIN_BATCHSIZE=32
TRAIN_KFOLD_SPLITS=8

# trenujemy model `n_splits` razy, w kazdym treningu mamy inne dane w treningu/walidacji; TODO: sprobowac zmienic liczbe treningow 6,10
def kfold_cross_validation(X, y, label_encoder, n_splits=5):
    """K-Fold cross validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_histories = []
    fold_scores = []
    all_y_true = []
    all_y_pred = []

    print("\n" + "=" * 70)
    print(f"ROZPOCZĘCIE {n_splits}-FOLD CROSS VALIDATION")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'=' * 70}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        #scaler = StandardScaler();
        #scaler.fit(X_train)
        #X_train = scaler.transform(X_train)
        #X_val = scaler.transform(X_val)

        print(f"Train set: {len(X_train)} próbek")
        print(f"Validation set: {len(X_val)} próbek")

        # Wizualizacja rozkładu klas w tym foldzie
        fig = plot_class_distribution(y_train, label_encoder,
                                      f"Rozkład klas - Fold {fold} (Train)")
        plt.savefig(f'class_distribution_fold_{fold}_train.png', dpi=300, bbox_inches='tight')
        #plt.show()

        fig = plot_class_distribution(y_val, label_encoder,
                                      f"Rozkład klas - Fold {fold} (Validation)")
        plt.savefig(f'class_distribution_fold_{fold}_val.png', dpi=300, bbox_inches='tight')
        #plt.show()

        # Obliczenie wag klas
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights_array))

        print("\nWagi klas:")
        for cls_idx, weight in class_weights.items():
            cls_name = label_encoder.classes_[cls_idx]
            print(f"  {cls_name}: {weight:.3f}")

        # Budowa i trening modelu
        MODEL_OBJECT = EmotionModels(
            input_shape=X_train.shape[1:],
            num_classes=len(np.unique(y))
        )
        #MODEL_OBJECT.build_light_cnn()# <<< -------------------- WYBOR MODELU SIECI NEURONOWEJ ---------------------
        #MODEL_OBJECT.build_deep_cnn()
        MODEL_OBJECT.build_resnet_inspired()
        #MODEL_OBJECT.inception_module()
        #MODEL_OBJECT.build_inception_inspired()
        #MODEL_OBJECT.attention_block()
        #MODEL_OBJECT.build_attention_cnn()
        #MODEL_OBJECT.build_mobilenet_inspired()
        #MODEL_OBJECT.build_crnn()
        
        
        
        MODEL_OBJECT.compile_model(
            optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LEARNRATE),
        )
        history = MODEL_OBJECT.train( # <<< --------------- HIPERPARAMETRY UCZENIA ------------------------
            X_train, y_train,
            X_val, y_val,
            epochs=TRAIN_EPOCHS,
            batch_size=TRAIN_BATCHSIZE
        )

        fold_histories.append(history)

        # Ewaluacja
        train_loss, train_acc = MODEL_OBJECT.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = MODEL_OBJECT.model.evaluate(X_val, y_val, verbose=0)

        y_pred = np.argmax(MODEL_OBJECT.model.predict(X_val, verbose=0), axis=1)

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        fold_scores.append({
            'fold': fold,
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'val_accuracy': val_acc,
            'val_loss': val_loss
        })

        print(f"\nWyniki Fold {fold}:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Macierz pomyłek dla tego foldu
        plot_confusion_matrix(y_val, y_pred, label_encoder.classes_, fold_num=fold)

        # Raport klasyfikacji
        print(f"\nClassification Report - Fold {fold}:")
        print(classification_report(y_val, y_pred,
                                    target_names=label_encoder.classes_,
                                    zero_division=0))

    # Podsumowanie wszystkich foldów
    print("\n" + "=" * 70)
    print("PODSUMOWANIE WSZYSTKICH FOLDÓW")
    print("=" * 70)

    avg_train_acc = np.mean([s['train_accuracy'] for s in fold_scores])
    avg_val_acc = np.mean([s['val_accuracy'] for s in fold_scores])
    std_train_acc = np.std([s['train_accuracy'] for s in fold_scores])
    std_val_acc = np.std([s['val_accuracy'] for s in fold_scores])

    print(f"\nŚrednia Train Accuracy: {avg_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Średnia Val Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")

    # Wizualizacja wyników
    plot_fold_results(fold_histories, fold_scores)

    # Łączna macierz pomyłek
    print("\n" + "=" * 70)
    print("ŁĄCZNA MACIERZ POMYŁEK ZE WSZYSTKICH FOLDÓW")
    print("=" * 70)
    plot_confusion_matrix(all_y_true, all_y_pred, label_encoder.classes_)

    print("\nŁączny Classification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=label_encoder.classes_,
                                zero_division=0))

    return fold_histories, fold_scores


def main():
    # Ścieżki do danych nEMO
    SAMPLES_PATH = "samples"
    TSV_FILE = "data.tsv"
    #SAMPLES_PATH = "nEMO-main/samples"
    #TSV_FILE = "nEMO-main/data.tsv"

    # Sprawdzenie czy używamy GPU
    gpus = tf.config.list_physical_devices('GPU')
    device_info = f"GPU ({gpus[0].name})" if gpus else "CPU"

    # 1. Ładowanie datasetu
    print("=" * 70)
    print(f"ŁADOWANIE DATASETU nEMO - Urządzenie: {device_info}")
    print("=" * 70)

    dataset = EmotionDataset(SAMPLES_PATH)
    X, y = dataset.load_dataset(TSV_FILE)

    print(f"\nKształt features: {X.shape}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print(f"Klasy emocji: {dataset.label_encoder.classes_}")

    # Wizualizacja rozkładu klas w całym datasecie
    fig = plot_class_distribution(y, dataset.label_encoder,
                                  "Rozkład klas - Cały dataset")
    plt.savefig('class_distribution_full_dataset.png', dpi=300, bbox_inches='tight')
    #plt.show()

    # 2. K-Fold Cross Validation
    fold_histories, fold_scores = kfold_cross_validation(
        X, y, dataset.label_encoder, n_splits=TRAIN_KFOLD_SPLITS
    )

    # 3. Zapisanie wyników
    results_df = pd.DataFrame(fold_scores)
    results_df.to_csv('kfold_results.csv', index=False)
    print("\nWyniki zapisane do 'kfold_results.csv'")

    return fold_histories, fold_scores, dataset.label_encoder


if __name__ == "__main__":
    fold_histories, fold_scores, label_encoder = main()