import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, MobileNetV2
import seaborn as sns

class EmotionModels:
    """Klasa zawierająca różne architektury modeli do rozpoznawania emocji"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None


    def build_light_cnn(self):
        """
        Lekki model CNN - szybki do treningu, dobry na początek
        Parametry: ~500K
        """
        model = keras.Sequential([
            # Blok 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            # Blok 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),


            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='Light_CNN')
        self.model = model
        return model

    def build_deep_cnn(self):
        """
        Głęboki model CNN - 4 bloki konwolucyjne
        Parametry: ~5M
        """
        model = keras.Sequential([
            # Blok 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Blok 4
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='Deep_CNN')
        self.model = model
        return model


    def build_resnet_inspired(self):
        """
        Model inspirowany ResNet z residual connections
        Parametry: ~3M
        """
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Residual Block 1
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        # Residual Block 2
        shortcut = layers.Conv2D(128, (1, 1), strides=2)(x)
        x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        # Residual Block 3
        shortcut = layers.Conv2D(256, (1, 1), strides=2)(x)
        x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        # Output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_Inspired')
        self.model = model
        return model


    def inception_module(self, x, filters):
        """Moduł Inception z równoległymi konwolucjami"""
        # Branch 1: 1x1
        branch1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

        # Branch 2: 1x1 -> 3x3
        branch2 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
        branch2 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 -> 5x5
        branch3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
        branch3 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(branch3)

        # Branch 4: MaxPooling -> 1x1
        branch4 = layers.MaxPooling2D((3, 3), strides=1, padding='same')(x)
        branch4 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(branch4)

        # Concatenate
        output = layers.Concatenate()([branch1, branch2, branch3, branch4])
        return output

    def build_inception_inspired(self):
        """
        Model inspirowany Inception z multi-scale feature extraction
        Parametry: ~4M
        """
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Inception modules
        x = self.inception_module(x, 32)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        x = self.inception_module(x, 64)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        # Output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='Inception_Inspired')
        self.model = model
        return model

    def attention_block(self, x, filters):
        """Blok uwagi (attention)"""
        # Query, Key, Value
        query = layers.Conv2D(filters, (1, 1))(x)
        key = layers.Conv2D(filters, (1, 1))(x)
        value = layers.Conv2D(filters, (1, 1))(x)

        # Attention weights
        attention = layers.Multiply()([query, key])
        attention = layers.Activation('softmax')(attention)

        # Apply attention
        output = layers.Multiply()([attention, value])
        output = layers.Add()([output, x])

        return output

    def build_attention_cnn(self):
        """
        CNN z mechanizmem uwagi (attention)
        Parametry: ~3.5M
        """
        inputs = layers.Input(shape=self.input_shape)

        # Conv blocks with attention
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 64)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

        # Output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='Attention_CNN')
        self.model = model
        return model

    def build_mobilenet_inspired(self):
        """
        Lekki model z depthwise separable convolutions
        Bardzo efektywny obliczeniowo
        Parametry: ~1M
        """
        model = keras.Sequential([
            # Initial conv
            layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu',
                          input_shape=self.input_shape),
            layers.BatchNormalization(),

            # Depthwise separable blocks
            layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.SeparableConv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.SeparableConv2D(512, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='MobileNet_Inspired')
        self.model = model
        return model

    # ============================================================================
    # MODEL 9: CRNN (Convolutional Recurrent Neural Network)
    # ============================================================================
    def build_crnn(self):
        """
        CRNN - CNN do ekstrakcji cech + RNN do sekwencji czasowych
        Parametry: ~2.5M
        """
        inputs = layers.Input(shape=self.input_shape)

        # CNN Feature Extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Reshape dla RNN - agregacja po wymiarze częstotliwości
        shape = x.shape
        x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

        # Bidirectional RNN
        x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.GRU(64))(x)
        x = layers.Dropout(0.4)(x)

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='CRNN')
        self.model = model
        return model
    def compile_model(self, optimizer='adam', learning_rate=0.001, loss='sparse_categorical_crossentropy'):
        """
        Kompilacja modelu z możliwością dostosowania hiperparametrów

        Args:
            optimizer: 'adam', 'sgd', 'rmsprop' lub obiekt optymalizera
            learning_rate: Współczynnik uczenia
            loss: Funkcja straty
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany! Najpierw użyj build_*() metody.")

        # Wybór optymalizera
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )

        print(f"✓ Model skompilowany z {optimizer}, learning_rate={learning_rate}")

    # ============================================================================
    # TRENING MODELU
    # ============================================================================
    def train(self, X_train, y_train, X_val, y_val,
              epochs=50,
              batch_size=32,
              early_stopping_patience=10,
              reduce_lr_patience=5,
              reduce_lr_factor=0.5,
              min_lr=1e-7,
              verbose=1):
        """
        Trening modelu z dostosowywalnymi hiperparametrami

        Args:
            X_train, y_train: Dane treningowe
            X_val, y_val: Dane walidacyjne
            epochs: Liczba epok
            batch_size: Rozmiar batcha
            early_stopping_patience: Cierpliwość dla early stopping
            reduce_lr_patience: Cierpliwość dla redukcji learning rate
            reduce_lr_factor: Współczynnik redukcji LR
            min_lr: Minimalna wartość learning rate
            verbose: Poziom szczegółowości (0, 1, 2)
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany i skompilowany!")

        print("\n" + "=" * 80)
        print("ROZPOCZĘCIE TRENINGU")
        print("=" * 80)
        print(f"Model: {self.model.name}")
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("=" * 80 + "\n")

        # Callbacks
        callbacks_list = []

        if early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)
            print(f"✓ Early Stopping włączony (patience={early_stopping_patience})")

        if reduce_lr_patience > 0:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
                verbose=1
            )
            callbacks_list.append(reduce_lr)
            print(f"✓ ReduceLROnPlateau włączony (patience={reduce_lr_patience}, factor={reduce_lr_factor})")

        # Model checkpoint (opcjonalnie)
        checkpoint = keras.callbacks.ModelCheckpoint(
            f'best_{self.model.name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0
        )
        callbacks_list.append(checkpoint)
        print(f"✓ ModelCheckpoint włączony (zapisywanie najlepszego modelu)")

        print("\n")

        # Trening
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )

        print("\n" + "=" * 80)
        print("TRENING ZAKOŃCZONY")
        print("=" * 80)

        # Podsumowanie wyników
        best_epoch = np.argmax(self.history.history['val_accuracy'])
        best_val_acc = self.history.history['val_accuracy'][best_epoch]
        best_train_acc = self.history.history['accuracy'][best_epoch]

        print(f"\nNajlepszy wynik:")
        print(f"  Epoka: {best_epoch + 1}")
        print(f"  Train Accuracy: {best_train_acc:.4f}")
        print(f"  Val Accuracy: {best_val_acc:.4f}")

        return self.history

    # ============================================================================
    # EWALUACJA MODELU
    # ============================================================================
    def evaluate(self, X_test, y_test, class_names=None, verbose=1):
        """
        Ewaluacja modelu na zbiorze testowym

        Args:
            X_test, y_test: Dane testowe
            class_names: Nazwy klas do wyświetlenia
            verbose: Poziom szczegółowości

        Returns:
            Dict z metrykami: loss, accuracy, predictions, classification_report
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany!")

        print("\n" + "=" * 80)
        print("EWALUACJA MODELU")
        print("=" * 80)

        # Podstawowa ewaluacja
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=verbose)

        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Predykcje
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Classification report
        if class_names is not None:
            print("\n" + "=" * 80)
            print("CLASSIFICATION REPORT")
            print("=" * 80)
            print(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_probs': y_pred_probs,
            'confusion_matrix': cm,
            'y_true': y_test
        }

        return results

    def predict(self, X, return_probs=False):
        """
        Predykcja dla nowych danych

        Args:
            X: Dane wejściowe
            return_probs: Czy zwrócić prawdopodobieństwa
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany!")

        probs = self.model.predict(X, verbose=0)

        if return_probs:
            return np.argmax(probs, axis=1), probs
        else:
            return np.argmax(probs, axis=1)

    def plot_history(self, save_path='training_history.png'):
        """Wizualizacja historii treningu"""
        if self.history is None:
            raise ValueError("Brak historii treningu! Najpierw wytrenuj model.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title(f'Model Accuracy - {self.model.name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Loss
        ax2.plot(self.history.history['loss'], label='Train', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title(f'Model Loss - {self.model.name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Historia treningu zapisana: {save_path}")
        plt.show()

    def plot_confusion_matrix(self, cm, class_names, save_path='confusion_matrix.png'):
        """Wizualizacja macierzy pomyłek"""
        plt.figure(figsize=(10, 8))

        # Normalizacja
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Normalized Count'})

        plt.title(f'Confusion Matrix - {self.model.name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Macierz pomyłek zapisana: {save_path}")
        plt.show()

    def save_model(self, filepath):
        """Zapisz model"""
        if self.model is None:
            raise ValueError("Model nie został zbudowany!")

        self.model.save(filepath)
        print(f"✓ Model zapisany: {filepath}")

    def load_model(self, filepath):
        """Załaduj model"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model załadowany: {filepath}")
