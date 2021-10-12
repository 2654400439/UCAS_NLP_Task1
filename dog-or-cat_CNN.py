import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
# ----------寻找最优超参数阶段----------
# 使用kerastuner库，进行超参数的自动学习。
# 首先定义函数用来传入超参数并返回模型
# 在此函数中构建神经网络，对于需要人工确定的超参数，使用kerastuner提供的接口进行替换
# 定义hyperband
# 利用search函数自动搜索最优超参数
hp = HyperParameters()
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(hp.Choice('num_filters_layer0',values=[16,64],default=16),(3,3),activation=tf.nn.relu,input_shape=(150,150,3)))
    model.add(keras.layers.MaxPool2D(2,2))
    for i in range(hp.Int("num_conv_layers",1,3)):
        model.add(keras.layers.Conv2D(hp.Choice('num_filters_layer'+str(i),values=[16,64],default=16),(3,3),activation=tf.nn.relu))
        model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int("hidden_units",128,512,step=32),activation=tf.nn.relu))
    model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
    model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.RMSprop(lr=0.001),metrics=['acc'])
    return model
# 使用验证集准确率作为超参数搜索的优化目标
tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=15,
    hyperparameters=hp,
    project_name='cat_or_dog_project'
)
# 定义搜索超参数阶段使用的图片generator
training_dir = "data/train/"
validation_dir = "data/val/"
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(training_dir,batch_size=50,class_mode='binary',target_size=(150,150))
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir,batch_size=50,class_mode='binary',target_size=(150,150))
# 使用kerastuner寻找最优参数，并利用最优参数构建模型
tuner.search(train_generator,epochs=15,validation_data=validation_generator)
best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)
model.summary()
# 得到的最优结构
# Conv2d 16个filters
# Maxpoolong2d
# Conv2d 16个filters
# Maxpooling2d
# Conv2d 64个filters
# Maxpooling2d
# Conv2d 16个filters
# Maxpooling2d
# Flatten
# Dense 352个神经元
# Dense 1个神经元

# ----------训练阶段----------
# 定义训练验证阶段使用的图片generator，对训练集增加数据增强操作
new_train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = new_train_datagen.flow_from_directory(training_dir,batch_size=100,class_mode='binary',target_size=(150,150))
new_validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = new_validation_datagen.flow_from_directory(validation_dir,batch_size=100,class_mode='binary',target_size=(150,150))
# 模型训练，并将训练结果保存在history变量中
history = model.fit_generator(train_generator,
                              epochs=150,
                              verbose=1,
                              validation_data=validation_generator)
# 保存训练好模型的所有参数
model.save('dog-or-cat_150epochs.h5')
# 绘制训练结果图
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))
plt.plot(epochs,acc,'r',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()



