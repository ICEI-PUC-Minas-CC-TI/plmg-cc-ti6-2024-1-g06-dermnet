import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, EarlyStopping

import time

# Configurações de ambiente
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Caminho para os dados
data_path = './train'

train_data = []
val_data = []

# Processamento dos dados de treinamento e validação
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    files = os.listdir(folder_path)

    numfile_used = int(0.2 * len(files))
    file_used = random.sample(files, numfile_used)

    num_train = int(0.85 * len(file_used))
    files_train = random.sample(file_used, num_train)
    files_val = list(set(file_used) - set(files_train))

    for file in files_train:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Erro ao carregar a imagem: {file_path}")
            continue
        img = cv2.resize(img, (224, 224))
        train_data.append((img, folder))

    for file in files_val:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Erro ao carregar a imagem: {file_path}")
            continue
        img = cv2.resize(img, (224, 224))
        val_data.append((img, folder))

# Construção do modelo
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

num_classes = 10
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Preparação dos dados
X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)

X_train = preprocess_input(np.array(X_train))
X_val = preprocess_input(np.array(X_val))

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

y_train_one_hot = to_categorical(y_train_encoded, num_classes)
y_val_one_hot = to_categorical(y_val_encoded, num_classes)

# Configurações de treinamento
EPOCHS = 20
BATCH_SIZE = 32

# Treinamento do modelo
start_time = time.time()
history = model.fit(X_train, y_train_one_hot, validation_data=(X_val, y_val_one_hot), epochs=EPOCHS, batch_size=BATCH_SIZE)
end_time = time.time()

execution_time = end_time - start_time
print("Tempo de treinamento: ", execution_time, " segundos")

# Salvar o modelo treinado
#model.save('./model_cpu.h5')

# Plotar os resultados
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training loss', marker='o')
plt.plot(epochs, val_loss, label='Validation loss', marker='o')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, train_acc, label='Training accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
