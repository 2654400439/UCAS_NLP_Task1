import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# # 将图片转为灰度图
train_cat_path = 'data/train/cat'
train_dog_path = 'data/train/dog'
val_cat_path = 'data/val/cat'
val_dog_path = 'data/val/dog'
filepath_list = [train_cat_path,train_dog_path,val_cat_path,val_dog_path]
for filepath in filepath_list:
    for file in os.listdir(filepath):
        img = Image.open(filepath + '/' + file)
        L = img.convert('L')
        L.save(filepath[:4] + '_gray' + filepath[4:] + '/' + file)

train_path = 'data_gray/train'
val_path = 'data_gray/val'

Training_data = []
Training_labels = []
for file in os.listdir(train_path+'/cat'):
    img = Image.open(train_path+'/cat/'+file)
    new = img.resize((150,150),Image.ANTIALIAS)
    Training_data.append(np.array(new).tolist())
    Training_labels.append(0)
for file in os.listdir(train_path+'/dog'):
    img = Image.open(train_path+'/dog/'+file)
    new = img.resize((150,150),Image.ANTIALIAS)
    Training_data.append(np.array(new).tolist())
    Training_labels.append(1)

Validation_data = []
Validation_labels = []
for file in os.listdir(val_path+'/cat'):
    img = Image.open(val_path+'/cat/'+file)
    new = img.resize((150,150),Image.ANTIALIAS)
    Validation_data.append(np.array(new).tolist())
    Validation_labels.append(0)
for file in os.listdir(val_path+'/dog'):
    img = Image.open(val_path+'/dog/'+file)
    new = img.resize((150,150),Image.ANTIALIAS)
    Validation_data.append(np.array(new).tolist())
    Validation_labels.append(1)

print("Training")
Training_data = np.array(Training_data)
Training_labels = np.array(Training_labels)
print(Training_data.shape)
print(Training_labels.shape)
print("Validation")
Validation_data = np.array(Validation_data)
Validation_labels = np.array(Validation_labels)
print(Validation_data.shape)
print(Validation_labels.shape)

index = list(range(len(Training_data)))
random.shuffle(index)
Training_data = Training_data[index]
Training_labels = Training_labels[index]
index = list(range(len(Validation_data)))
random.shuffle(index)
Validation_data = Validation_data[index]
Validation_labels = Validation_labels[index]

model = keras.Sequential()
model.add(keras.layers.LSTM(64, input_shape=(150,150), return_sequences=True))
model.add(keras.layers.LSTM(64, return_sequences=True))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['acc'])

history = model.fit(
    Training_data,
    Training_labels,
    epochs=150,
    batch_size=64,
    validation_data=(Validation_data,Validation_labels),
    verbose=1
          )

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
