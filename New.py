import pandas as pd
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import kagglehub
import warnings

warnings.filterwarnings("ignore")

# Download datasets
print("Downloading datasets...")
ravdess_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("RAVDESS path:", ravdess_path)

crema_path = kagglehub.dataset_download("ejlok1/cremad")
print("CREMA path:", crema_path)

tess_path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("TESS path:", tess_path)

savee_path = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")
print("SAVEE path:", savee_path)

# Data paths
ravdess = os.path.join(ravdess_path, "audio_speech_actors_01-24")
Crema = os.path.join(crema_path, "AudioWAV")
Tess = os.path.join(tess_path, "tess toronto emotional speech set data", "TESS Toronto emotional speech set data")
Savee = os.path.join(savee_path, "ALL")
print("Processing RAVDESS...")
# Process RAVDESS
file_emotion = []
file_path = []
for i in os.listdir(ravdess):
    actor_path = os.path.join(ravdess, i)
    if not os.path.isdir(actor_path):
        continue
    actor = os.listdir(actor_path)
    for f in actor:
        if not f.endswith('.wav'):
            continue
        part = f.split('.')[0].split('-')
        file_emotion.append(int(part[2]))
        file_path.append(os.path.join(actor_path, f))

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
ravdess_df = pd.concat([emotion_df, path_df], axis=1)
ravdess_df.Emotions.replace(
    {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
    inplace=True)
print("Processing CREMA...")
# Process CREMA
crema_directory_list = os.listdir(Crema)
file_emotion = []
file_path = []
for file in crema_directory_list:
    if not file.endswith('.wav'):
        continue
    file_path.append(os.path.join(Crema, file))
    part = file.split('_')
    emotion_map = {'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral'}
    file_emotion.append(emotion_map.get(part[2], 'Unknown'))

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
print("Processing TESS...")
# Process TESS
tess_directory_list = os.listdir(Tess)
file_emotion = []
file_path = []
for dir in tess_directory_list:
    dir_path = os.path.join(Tess, dir)
    if not os.path.isdir(dir_path):
        continue
    directories = os.listdir(dir_path)
    for file in directories:
        if not file.endswith('.wav'):
            continue
        part = file.split('.')[0].split('_')[2]
        file_emotion.append('surprise' if part == 'ps' else part)
        file_path.append(os.path.join(dir_path, file))

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
print("Processing SAVEE...")
# Process SAVEE
savee_directory_list = os.listdir(Savee)
file_emotion = []
file_path = []
for file in savee_directory_list:
    if not file.endswith('.wav'):
        continue
    file_path.append(os.path.join(Savee, file))
    part = file.split('_')[1]
    ele = part[:-6]
    emotion_map = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy', 'n': 'neutral', 'sa': 'sad'}
    file_emotion.append(emotion_map.get(ele, 'surprise'))

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)

# Combine datasets
data_path = pd.concat([ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)


# Data augmentation functions
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)


# Feature extraction
def zcr(data, frame_length, hop_length):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))


def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))


def mfcc(data, sr, frame_length=2048, hop_length=512):
    mfcc_feat = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc_feat.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.hstack((zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result


def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data)
    audio = np.array(aud)

    noised_audio = noise(data)
    aud2 = extract_features(noised_audio)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitch(data, sr)
    aud3 = extract_features(pitched_audio)
    audio = np.vstack((audio, aud3))

    pitched_noised_audio = noise(pitch(data, sr))
    aud4 = extract_features(pitched_noised_audio)
    audio = np.vstack((audio, aud4))

    return audio

print('Extracting features...')
# Extract features from all files
X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    features = get_features(path)
    for i in features:
        X.append(i)
        Y.append(emotion)

# Prepare data
X = np.array(X)
X = np.nan_to_num(X, nan=0.0)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)


# PyTorch Dataset
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(2)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = EmotionDataset(x_train, y_train)
test_dataset = EmotionDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# CNN Model in PyTorch
class EmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes=7):
        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout4 = nn.Dropout(0.2)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout5 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(128 * self._get_conv_output(input_size), 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output(self, input_size):
        x = torch.zeros(1, 1, input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))
        return x.shape[2]

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout4(self.pool4(torch.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(self.pool5(torch.relu(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn6(self.fc1(x)))
        x = self.fc2(x)

        return x


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
model = EmotionCNN(input_size=x_train.shape[1], num_classes=7).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=0.00001)

# Training
num_epochs = 2
best_acc = 0.0
print("Training")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / len(train_dataset)

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / len(test_dataset)
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_pytorch.pth')

    print(f'Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# Evaluation
model.load_state_dict(torch.load('best_model_pytorch.pth'))
model.eval()

y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.numpy())

y_pred = label_encoder.inverse_transform(y_pred)
y_true = label_encoder.inverse_transform(y_true)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_true, y_pred))

# Save preprocessing objects
import pickle

with open('scaler_pytorch.pickle', 'wb') as f:
    pickle.dump(scaler, f)
with open('encoder_pytorch.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)


# Prediction function
def predict_emotion(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data).reshape(1, -1)
    features = scaler.transform(features)
    features = torch.FloatTensor(features).unsqueeze(2).to(device)

    model.eval()
    with torch.no_grad():
        output = model(features)
        _, predicted = torch.max(output, 1)

    emotion = label_encoder.inverse_transform(predicted.cpu().numpy())
    return emotion[0]