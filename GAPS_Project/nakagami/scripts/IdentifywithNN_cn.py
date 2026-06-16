# 导入库 -------------------------------------------------------------
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import *
import random

import shutil
from pathlib import Path
import pathlib
import glob
import multiprocessing
import math

import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D, Add
from tensorflow.keras.layers import BatchNormalization, Input, concatenate, GlobalAveragePooling3D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, Adadelta, Adamax, Nadam, RMSprop
from keras.utils import np_utils, plot_model, to_categorical
from keras.utils import Sequence
from keras import initializers
#from tensorflow.contrib.tpu.python.tpu import keras_support
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.callbacks import History
from keras.models import model_from_json
from tensorflow.keras import regularizers
import os.path

import csv
import os
import time

import tensorflow as tf
from tensorflow.keras import backend as K
from decimal import Decimal, ROUND_HALF_UP
# -------------------------------------------------------------------

# 指定保存学习数据的路径
download_path = "/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized/shuffled/"
base_dir = Path(download_path)

# 用于输入学习数据的 Generator
class MyGenerator(Sequence):
  def __init__(self):
    self.reset()

  def reset(self):
    self.edep_gene = []
    self.edep_tof_gene = []
    self.tof_gene = []
    self.labels_gene = []
    #self.mode_gene = []

  def flow_from_directory(self, directory, batch_size): # 使用 generator 获取数据
    while True:
      file_list = sorted(pathlib.Path(directory).glob("*.csv")) # 获取 directory 内的 csv 文件列表
      data_list = []
      numofevents_flow = 0
      for path in file_list:
        print(path)
        with open(path) as f: # 打开 csv 文件
          reader = csv.reader(f, delimiter=",") # 读取 csv 文件内容
          for row in reader: # 逐行读取
            targets_label = row[2]  # 第3列是粒子标签（pbar:0, dbar:1）
            #targets_mode = row[3]  # 第4列是衰变模式标签（atrest:0, inflight:1）
            #targets_categorical = [targets_label, targets_mode] # 粒子种类和衰变模式的复合标签
            self.labels_gene.append(targets_label) # 保存粒子种类标签

            # 声明用于保存各类信息的 vector
            edep_by_event = []    # Si wafer 中的能量损失
            tof_by_event = []     # TOF counter 中与入射粒子相关的物理量
            edep_eachPaddle = []  # 各 TOF paddle 中的能量损失

            # 作为学习用数据进行读取
            for i in range(len(row)):
              if 2 < i < 1443:      # Si wafer 中的能量损失（实际为 0-indexed 的 col 3~1442）
                if len(row[i]) == 0: # 不知为何包含 NULL，可能是 shuffle.py 有问题
                  row[i] = 0
                signal_Si = float(row[i])
                edep_by_event.append(signal_Si)

              if 1442 < i < 1452:  # 各 TOF paddle 中的能量损失（实际为 0-indexed 的 col 1443~1451，共9维）
                signal_TOFE = float(row[i])
                edep_eachPaddle.append(signal_TOFE)

              #if 1617 < i < 1629:   # primary 的物理量（这部分被注释掉，实际没有使用）
              #  signal_TOFB = float(row[i])
              #  tof_by_event.append(signal_TOFB)

            edep_by_event_array = np.array(edep_by_event, dtype=np.float32) # 转换为 numpy 数组
            edep_by_event_re = np.reshape(edep_by_event_array, (10, 12, 12, 1)) # 转换为 10×12×12×1 的三维输入形式
            edep_eachPaddle_array = np.array(edep_eachPaddle, dtype=np.float32)
            #tof_by_event_array = np.array(tof_by_event, dtype=np.float32)

            # 保存各类信息
            self.edep_gene.append(edep_by_event_re)
            self.edep_tof_gene.append(edep_eachPaddle)
            #self.tof_gene.append(tof_by_event_array)

            if numofevents_flow == 0:
              #print(edep_by_event_re[0])
              #print(tof_by_event_array)
              numofevents_flow += 1

            if len(self.labels_gene) == batch_size: # 累积到 batch_size 个事件后返回
              #targets_categorical = to_categorical(self.labels_gene, num_classes=4)#, dtype='float32')
              targets_np = np.array(self.labels_gene, dtype=np.int8)  # 正解数据（粒子种类标签）
              input1 = np.array(self.edep_gene, dtype=np.float32)     # 输入1：Si wafer 中的能量损失
              input2 = np.array(self.edep_tof_gene, dtype=np.float32) # 输入2：各 TOF paddle 中的能量损失
              #input3 = np.array(self.tof_gene, dtype=np.float32)      # 输入3：primary 物理量候选，但实际没有使用
              self.reset() # 初始化 *_gene 的内容
              yield [input1, input2], targets_np # 不是 return，而是 yield；返回后继续处理，读完所有文件后结束

  def for_results(self, directory, numofevents): # 记录结果时使用，获取预测事件的识别ID和粒子种类
    aa = 0
    while True:
      if aa > numofevents - 1:
        break
      file_list = sorted(pathlib.Path(directory).glob("*.csv"))
      for path in file_list:
        if aa > numofevents - 1:
          break
        with open(path) as f:
          reader = csv.reader(f, delimiter=",")
          for row in reader:
            if aa > numofevents - 1:
              break
            yield (row[0]), int(row[1]), int(float(row[2]))
            aa += 1

def save_Model(json, model_weights):
  save_directory = 'Models/' # 保存目录，需要事先创建
  print('## Save the architecture of a model ')
  json_string = model.to_json()
  open(os.path.join(json), 'w').write(json_string)
  model.save_weights(model_weights, save_format='h5')
  print('## Save weights ')
  shutil.copy(json, save_directory)
  shutil.copy(model_weights, save_directory)

def output_Results(dire, numofdata, file_name):
  print("## Predict each Particle ##")
  output_file = file_name # 输出文件名
  prediction_generator = MyGenerator()
  labels_from_generator = prediction_generator.for_results(dire, numofdata)
  results = model.predict_generator(
      prediction_generator.flow_from_directory(dire, batch_size),
      steps=int(numofdata / batch_size),
      verbose=0
  )
  print(results)

  results_list = results.tolist()
  iterator = 0
  label_p = []
  for n in labels_from_generator:
    if iterator == numofdata:
      break
    label_p.append(n)
    iterator += 1

  output_string = []
  for i in range(numofdata):
    output_string.append(str(label_p[i][0]) + ' ' +  # 文件ID
                         str(label_p[i][1]) + ' ' +  # event ID
                         str(label_p[i][2]) + ' ' +  # 实际粒子种类标签
                         str(results_list[i]) + '\n')# 预测得到的“像反重氘核”的程度

  file_output = open(output_file, 'w')
  file_output.writelines(output_string)
  file_output.close()

###########################################################
basic_name = '220322_for4M' # 生成模型名称的基础字符串，相当于本次学习的识别 key
json_name = basic_name + '.json'
model_name = basic_name + '.hdf5'
###########################################################

#
# 学习模型开始
#
print('###### Start Learning ######')

numoftraining_data  = 1600000 # 学习用数据数量
numofvalidation_data = 400000 # 验证用数据数量
batch_size = 200              # 一次学习中处理的数据数量
epochs = 50                   # 学习次数上限
learning_rate = 0.00004       # 学习率
#L2_regularizer = 0.001
activation_function = 'relu'  # 激活函数
loss_function = 'binary_crossentropy' # 损失函数
monitor = 'val_accuracy'
train_directory = download_path + "train_5cross" # 存放学习用数据的目录名
val_directory = download_path + "valid_5cross"   # 存放评价用数据的目录名

# 学习数据输入时的形状
input_shape1 = (10, 12, 12, 1) # Si(Li) 3D energy map
input_shape2 = (9)             # TOF paddle energy 9维
input_shape3 = (11)            # primary 物理量 11维候选，但实际没有使用

## optimizer 一览 ##

#opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#opt = Adagrad(lr=0.0005, epsilon=None, decay=0.0)
#opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#opt = Adadelta(lr=0.1, rho=0.95, epsilon=None, decay=0.0)
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#opt = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

## optimizer 参数说明 ##
# SGD(...)
# Adagrad(...)  # default
# RMSprop(...)  # default，Adagrad 的改良版
# Adadelta(...) # default，Adagrad / RMSprop 的改良版
# Adam(...)     # Adagrad / RMSprop / Adadelta 的改良版
# Adamax(...)
# Nadam(...)    # default

# PreActive 型 Residual Block 的结构
def ResBlock_bottleNeck_PreActive(input_, num_filters, output_filters):
  shortcut = input_
  res = BatchNormalization()(input_)
  res = Activation(activation=activation_function)(res)

  res = Conv3D(filters=num_filters, kernel_size=(1, 1, 1), padding="same")(res)
  res = BatchNormalization()(res)
  res = Activation(activation=activation_function)(res)
  res = Dropout(rate=0.1)(res)
  res = Conv3D(filters=num_filters, kernel_size=(3, 3, 3), padding="same")(res)
  res = BatchNormalization()(res)
  res = Activation(activation=activation_function)(res)
  res = Conv3D(filters=output_filters, kernel_size=(1, 1, 1), padding="same")(res)
  res = Add()([res, shortcut])
  return res

# 使用 functional API 构建学习模型
def build_model_CNN(input_shape1, input_shape2):
  input1 = Input(shape=input_shape1)
  input2 = Input(shape=input_shape2)
  #input3 = Input(shape=input_shape3)

  # 以 Si(Li) 中的能量损失为输入的分支
  x = BatchNormalization()(input1)
  x = Activation(activation=activation_function)(x)
  x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding="same")(x)

  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = MaxPooling3D(pool_size=(2, 2, 2))(x)

  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = MaxPooling3D(pool_size=(2, 2, 2))(x)

  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = GlobalAveragePooling3D(data_format=None)(x)
  x = Dense(256)(x)
  x = BatchNormalization()(x)
  x = Activation(activation=activation_function)(x)
  model_Asitis = Model(inputs=input1, outputs=x)

  # 以各 TOF paddle 中的能量损失为输入的分支
  y = Dense(256)(input2)
  y = BatchNormalization()(y)
  y = Activation(activation=activation_function)(y)
  y = Dense(256)(y)
  y = BatchNormalization()(y)
  y = Activation(activation=activation_function)(y)
  #y = Dropout(rate=0.2)(y)
  y = Dense(128)(y)
  y = BatchNormalization()(y)
  y = Activation(activation=activation_function)(y)
  y = Dense(64)(y)
  y = BatchNormalization()(y)
  y = Activation(activation=activation_function)(y)
  model_second = Model(inputs=input2, outputs=y)

  '''
  # 以入射粒子相关物理量为输入的分支
  # 注意：这一整段被注释掉，实际没有使用。
  z = Dense(64)(input3)
  z = BatchNormalization()(z)
  z = Activation(activation=activation_function)(z)
  z = Dense(64)(z)
  z = BatchNormalization()(z)
  z = Activation(activation=activation_function)(z)
  z = Dropout(rate=0.2)(z)
  z = Dense(64)(z)
  z = BatchNormalization()(z)
  z = Activation(activation=activation_function)(z)
  z = Dense(64)(z)
  z = BatchNormalization()(z)
  z = Activation(activation=activation_function)(z)
  model_third = Model(inputs=input3, outputs=z)

  # 合并 model_second 和 model_third
  combined_yz = concatenate([model_second.output, model_third.output])
  combined_yz = Dense(64)(combined_yz)
  combined_yz = BatchNormalization()(combined_yz)
  combined_yz = Activation(activation=activation_function)(combined_yz)
  model_tof = Model(inputs=[model_second.input, model_third.input], outputs=combined_yz)
  '''

  # 合并 Si(Li) CNN 分支和 TOF paddle energy DNN 分支
  combined = concatenate([model_Asitis.output, model_second.output])#, axis=4)
  final = Dense(128)(combined)
  final = BatchNormalization()(final)
  final = Activation(activation=activation_function)(final)
  #final = Dropout(rate=0.2)(final)
  final = Dense(128)(final)
  final = BatchNormalization()(final)
  final = Activation(activation=activation_function)(final)
  final = Dense(64)(final)
  final = BatchNormalization()(final)
  final = Activation(activation=activation_function)(final)
  final = Dense(1, activation="sigmoid")(final)

  model_return = Model(inputs=[model_Asitis.input, model_second.input], outputs=final)
  model_return.summary()
  return model_return

#tf.keras.backend.clear_session()
early_stopping = EarlyStopping(patience=4, verbose=1, monitor=monitor) # Keras 的防止过学习功能

# EarlyStopping 参数说明：
# monitor：监视的值
# min_delta：判断为改善所需的最小变化量。若变化绝对值小于 min_delta，则认为没有改善。
# patience：指定多少个 epoch 内没有改善时停止训练。
# verbose：输出详细程度。
# mode：auto / min / max。
# min 模式下，当监视值停止下降时结束训练。
# max 模式下，当监视值停止上升时结束训练。
# auto 模式下，会根据监视值自动推定趋势。

print("## Built Model ")

# 多 GPU 计算所需
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
  model = build_model_CNN(input_shape1, input_shape2) # 声明用于学习的模型
  tf.keras.utils.plot_model(model, to_file='220319_Model.png', show_layer_names=False, show_shapes=True)
  model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
print("## Model Compiled ")

train_generator = MyGenerator()
validation_generator = MyGenerator()

# 执行学习
history = model.fit_generator(generator=train_generator.flow_from_directory(train_directory, batch_size)
                             ,steps_per_epoch=int(numoftraining_data / batch_size)
                             ,verbose=1
                             ,epochs=epochs
                             ,max_queue_size=20
                             ,validation_data=validation_generator.flow_from_directory(val_directory, batch_size)
                             ,validation_steps=int(numofvalidation_data / batch_size)
                             ,callbacks=[early_stopping]#,tensorboard_callback]
                            )

# 保存学习结果
save_Model(json_name, model_name)

# 使用训练好的模型对评价数据进行预测时的分数（损失函数输出和 Accuracy）
score = model.evaluate_generator(
    validation_generator.flow_from_directory(val_directory, batch_size),
    steps=int(numofvalidation_data / batch_size),
    verbose=0
)
print('## Test loss:', score[0])
print('## Test accuracy:', score[1])

#plot_history(history) # 显示各种结果

print('###### End Learning ######')

#
# 学习模型结束
#

print(history.history['accuracy'])
train_acc_list = history.history['accuracy']
train_loss_list = history.history['loss']
print(train_acc_list)
print(train_loss_list)
print(history.history['val_accuracy'])
val_acc_list = history.history['val_accuracy']
val_loss_list = history.history['val_loss']
print(val_acc_list)
print(val_loss_list)


## 创建学习精度曲线用的数据
## 将 epoch 数对应的学习精度保存为 csv 文件

train_accuracy_csv = basic_name + '_train_validation_accuracy_loss.csv'
f = open(train_accuracy_csv, 'a')
csvWriter_history = csv.writer(f)
for i in range(len(train_acc_list)):
  list_csv = []
  list_csv.append(i)
  list_csv.append(train_acc_list[i])
  list_csv.append(val_acc_list[i])
  list_csv.append(train_loss_list[i])
  list_csv.append(val_loss_list[i])
  csvWriter_history.writerow(list_csv)
f.close()


## 使用评价用数据输出预测精度
## 下面将正确 label 和模型预测结果输出为 text 文件

output_file = basic_name + '_label_output.dat'
comfirm_file = basic_name + '_comfirm_training.dat'
output_Results(val_directory, numofvalidation_data, output_file) # 保存评价用数据的结果
output_Results(train_directory, numoftraining_data, comfirm_file) # 对训练中使用的数据也保存结果

print("### Complete ###")