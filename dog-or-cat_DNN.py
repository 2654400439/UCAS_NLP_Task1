import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import os
# 将图片转为灰度图
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
# 构建全连接神经网络模型
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(150,150)))
model.add(keras.layers.Dense(512,activation=tf.nn.relu))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(64,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001),loss=tf.losses.binary_crossentropy,metrics=['acc'])
# 定义样本生成器
training_dir = "data_gray/train/"
validation_dir = "data_gray/val/"
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(training_dir,batch_size=100,class_mode='binary',color_mode='grayscale',target_size=(150,150))
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir,batch_size=100,class_mode='binary',color_mode='grayscale',target_size=(150,150))
# 训练模型
history = model.fit_generator(train_generator,
                              epochs=200,
                              verbose=1,
                              validation_data=validation_generator)
# 保存模型
model.save('dog-or-cat-DNN_20   0epochs.h5')
# 绘制模型训练准确率变化图
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))
plt.plot(epochs,acc,'r',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()




