
# ライブラリをインポート-------------------------------------------------------------
import pandas as pd
from pandas import Series,DataFrame

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
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv3D,MaxPooling3D,Add
from tensorflow.keras.layers import BatchNormalization,Input,concatenate,GlobalAveragePooling3D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam,Adagrad,SGD,Adadelta,Adamax,Nadam,RMSprop
from keras.utils import np_utils,plot_model,to_categorical
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
#---------------------------------------------------------------------------------

# 学習データが保存されているパスを指定
download_path = "/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized/shuffled/"
base_dir = Path(download_path)

# 学習データ入力のためのGenerator
class MyGenerator(Sequence):
  def __init__(self):
    self.reset()

  def reset(self):
    self.edep_gene = []
    self.edep_tof_gene = []
    self.tof_gene = []
    self.labels_gene = []
    #self.mode_gene = []

  def flow_from_directory(self, directory, batch_size): #generatorを使ったデータの取得
    while True:
      file_list = sorted(pathlib.Path(directory).glob("*.csv")) #'directory'内のcsvfileをlistとして取得
      data_list = []
      numofevents_flow = 0
      for path in file_list:
        print(path)
        with open(path) as f: #csvfileを開く
          reader = csv.reader(f,delimiter=",") #csvfileの中身を読み込む
          for row in reader: #forで一行ずつループを回す。
            targets_label = row[2]  #3列目は粒子ラベル(pbar:0, dbar:1)
            #targets_mode = row[3]  #4列目は崩壊モードラベル(atrest:0, inflight:1)
            #targets_categorical = [targets_label, targets_mode] #粒子種と崩壊モードの複合ラベル
            self.labels_gene.append(targets_label) #粒子種ラベルを保存
            # 各情報を保存するvectorの宣言
            edep_by_event = []    #Siウエハでのエネルギー損失
            tof_by_event = []     #TOFカウンタでの入射粒子に関する物理量
            edep_eachPaddle = []  #各TOFパドルにおけるエネルギー損失
            #学習用のデータとして取得
            for i in range(len(row)):
              if 2 < i < 1443:      #6~1445列は各Siウエハでのエネルギー損失
                if len(row[i])==0: # なぜかNULLが含まれているため、shuffle.pyに問題ある可能性あり
                  row[i]=0
                signal_Si = float(row[i])
                edep_by_event.append(signal_Si)
              if 1442 < i < 1452:  #1446~1617は各TOFパドルでのエネルギー損失
                signal_TOFE = float(row[i])
                edep_eachPaddle.append(signal_TOFE)
              #if 1617 < i < 1629:   #1618~1628はprimaryの物理量
              #  signal_TOFB = float(row[i])
              #  tof_by_event.append(signal_TOFB)
            edep_by_event_array = np.array(edep_by_event, dtype=np.float32) #numpy配列に変換
            edep_by_event_re = np.reshape(edep_by_event_array,(10,12,12,1)) #12*12*10層(3次元配列)の形に変換
            edep_eachPaddle_array = np.array(edep_eachPaddle, dtype=np.float32)
            #tof_by_event_array = np.array(tof_by_event, dtype=np.float32)
            #各情報を保存
            self.edep_gene.append(edep_by_event_re)
            self.edep_tof_gene.append(edep_eachPaddle)
            #self.tof_gene.append(tof_by_event_array)
            
            if numofevents_flow == 0:
              #print(edep_by_event_re[0])
              #print(tof_by_event_array)
              numofevents_flow+=1
            if len(self.labels_gene) == batch_size: #'batch_size'分たまったら返す
              #targets_categorical = to_categorical(self.labels_gene, num_classes=4)#, dtype='float32')
              targets_np = np.array(self.labels_gene, dtype=np.int8)  #正解データ(粒子種ラベル)
              input1 = np.array(self.edep_gene, dtype=np.float32)     #入力その1(Siウエハでのエネルギー損失)
              input2 = np.array(self.edep_tof_gene, dtype=np.float32) #入力その2(TOFカウンタでの入射粒子に関する物理量)
              #input3 = np.array(self.tof_gene, dtype=np.float32)      #入力その3(各TOFパドルにおけるエネルギー損失)
              self.reset() #*_geneの中身を初期化
              yield [input1,input2], targets_np #returnではなくyieldで返して処理を続行。全てのファイルを読み込んだら自動的に終了する
            
  def for_results(self, directory, numofevents):#結果を記す際に使用、予想したイベントの識別ID,粒子種を取得
    aa = 0
    while True:
      if aa > numofevents-1:
        break
      file_list = sorted(pathlib.Path(directory).glob("*.csv"))
      for path in file_list:
        if aa > numofevents-1:
          break
        with open(path) as f:
          reader = csv.reader(f,delimiter=",")
          for row in reader:
            if aa > numofevents-1:
              break
            yield (row[0]),int(row[1]),int(float(row[2]))
            aa += 1

def save_Model(json,model_weights):
  save_directory = 'Models/' #保存先ディレクトリ,予め作成する必要あり!
  print('## Save the architecture of a model ')
  json_string = model.to_json()
  open(os.path.join(json),'w').write(json_string)
  model.save_weights(model_weights, save_format='h5')
  print('## Save weights ')
  shutil.copy(json,save_directory)
  shutil.copy(model_weights,save_directory)

def output_Results(dire,numofdata,file_name):
  print("## Predict each Particle ##")
  output_file = file_name # 出力ファイル名
  prediction_generator = MyGenerator()
  labels_from_generator = prediction_generator.for_results(dire, numofdata)
  results = model.predict_generator(prediction_generator.flow_from_directory(dire,batch_size), steps=int(numofdata/batch_size), verbose=0)
  print(results)

  results_list = results.tolist()
  iterator = 0
  label_p = []
  for n in labels_from_generator:
    if iterator == numofdata:
      break
    label_p.append(n)
    iterator+=1

  output_string = []
  for i in range(numofdata):
    output_string.append(str(label_p[i][0]) + ' ' +  #ファイルID
                         str(label_p[i][1]) + ' ' +  #イベントID
                         str(label_p[i][2]) + ' ' +  #実際の粒子種ラベル
                         str(results_list[i]) + '\n')#予想した反重陽子らしさ

  file_output = open(output_file,'w')
  file_output.writelines(output_string)
  file_output.close()

###########################################################
basic_name = '220322_for4M' #作成されるモデルの名前のベース,学習に対する識別キーのようなもの
json_name = basic_name + '.json'
model_name = basic_name + '.hdf5'
###########################################################

#
# learning model, start
#
print('###### Start Learning ######')

numoftraining_data  =  1600000 # 学習用に使うデータの数
numofvalidation_data =  400000 # 検証用に使うデータの数
batch_size = 200               # 一度の学習で扱うデータの数
epochs = 50                    # 学習回数の上限
learning_rate = 0.00004        # 学習率
#L2_regularizer=0.001 
activation_function='relu'     # 活性化関数
loss_function='binary_crossentropy' # 損失関数
monitor='val_accuracy'
train_directory = download_path + "train_5cross" #学習用データが収容されてるディレクトリ名
val_directory = download_path + "valid_5cross"   #評価用データが収容されてるディレクトリ名

#学習データの入力時の形
input_shape1 = (10,12,12,1)
input_shape2 = (9)
input_shape3 = (11)

## optimizer一覧 ##

#opt=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#opt=Adagrad(lr=0.0005, epsilon=None, decay=0.0)
#opt=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#opt=Adadelta(lr=0.1, rho=0.95, epsilon=None, decay=0.0)
opt=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#opt=Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#opt=tf.train.RMSPropOptimizer(learning_rate=0.001)

## optimizerの引数 ##
#SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#Adagrad(lr=0.01, epsilon=None, decay=0.0) ##default
#RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) ##default (Adagrad の改良版)
#Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0) ##default (Adagrad, RMSprop の改良版)
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) (Adagrad, RMSprop, Adadelts の改良版)
#Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004) ##default

# PreActive型の Residual Block の構造
def ResBlock_bottleNeck_PreActive(input_, num_filters, output_filters):
  shortcut = input_
  res = BatchNormalization()(input_)
  res = Activation(activation=activation_function)(res)
  res = Conv3D(filters=num_filters,kernel_size=(1,1,1),padding="same")(res)

  res = BatchNormalization()(res)
  res = Activation(activation=activation_function)(res)
  res = Dropout(rate=0.1)(res)
  res = Conv3D(filters=num_filters,kernel_size=(3,3,3),padding="same")(res)

  res = BatchNormalization()(res)
  res = Activation(activation=activation_function)(res)
  res = Conv3D(filters=output_filters,kernel_size=(1,1,1),padding="same")(res)
  res = Add()([res, shortcut])
  return res
  

# 学習モデルの構築 by functional API
def build_model_CNN(input_shape1, input_shape2):
  input1 = Input(shape=input_shape1)
  input2 = Input(shape=input_shape2)
  #input3 = Input(shape=input_shape3)

  #SILIにおけるエネルギー損失を入力とする枝
  x = BatchNormalization()(input1)
  x = Activation(activation=activation_function)(x)
  x = Conv3D(filters=512,kernel_size=(3,3,3),padding="same")(x)
  
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = MaxPooling3D(pool_size=(2,2,2))(x)

  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = MaxPooling3D(pool_size=(2,2,2))(x)

  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)
  x = ResBlock_bottleNeck_PreActive(x, 64, 512)

  x = GlobalAveragePooling3D(data_format=None)(x)
  x = Dense(256)(x)
  x = BatchNormalization()(x)
  x = Activation(activation=activation_function)(x)
  model_Asitis = Model(inputs=input1, outputs=x)
  
  # 各TOFパドルにおけるエネルギー損失を入力とする枝
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
  #入射粒子に関する物理量を入力とする枝
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

  # model_secondとmodel_thirdを合体
  combined_yz = concatenate([model_second.output, model_third.output])
  combined_yz = Dense(64)(combined_yz)
  combined_yz = BatchNormalization()(combined_yz)
  combined_yz = Activation(activation=activation_function)(combined_yz)
  model_tof = Model(inputs=[model_second.input, model_third.input], outputs=combined_yz)
  '''
  # model_Asitisとmodel_tofを合体
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
early_stopping = EarlyStopping(patience=4, verbose=1, monitor=monitor) # 過学習を防止するためのKeras機能

#EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='auto')
#monitor:監視する値
#min_delta:監視する値について改善として判定される最小変化値。つまり、min_deltaよりも絶対値の変化が小さければ改善していないとみなす。
#patience:ここで指定したエポック数の間（監視する値に）改善がないと、訓練が停止する。
#verbose:冗長モード
#mode:{auto,min,max}の内、一つが選択される。minモードでは、監視する値の減少が停止した際に、訓練を終了。maxモードでは、監視する値の増加が停止した際に、訓練を終了。autoモードでは、この傾向は自動的に監視されている値から推定する。

print("## Built Model ")

# マルチGPU計算のために必要らしい
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
  model = build_model_CNN(input_shape1,input_shape2) #学習に用いるモデルの宣言
  tf.keras.utils.plot_model(model, to_file='220319_Model.png', show_layer_names=False, show_shapes=True)
  model.compile(optimizer=opt,loss=loss_function,metrics=['accuracy'])
print("## Model Compiled ")

train_generator = MyGenerator()
validation_generator = MyGenerator()

#学習実行
history = model.fit_generator(generator=train_generator.flow_from_directory(train_directory,batch_size)
                             ,steps_per_epoch=int(numoftraining_data/batch_size)
                             ,verbose=1
                             ,epochs=epochs
                             ,max_queue_size=20
                             ,validation_data=validation_generator.flow_from_directory(val_directory,batch_size)
                             ,validation_steps=int(numofvalidation_data/batch_size)
                             ,callbacks=[early_stopping]#,tensorboard_callback]
                            )
#学習結果の保存
save_Model(json_name,model_name)

#学習したモデルを用いて評価用データに対する予測を行った際のスコア(損失関数の出力,Accuracy)
score = model.evaluate_generator(validation_generator.flow_from_directory(val_directory,batch_size), steps=int(numofvalidation_data/batch_size), verbose=0)
print('## Test loss:', score[0])
print('## Test accuracy:', score[1])

#plot_history(history) #色々と結果を表示

print('###### End Learning ######')

#
# learning model, end
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


## 学習精度曲線用のデータ作成
## エポック数に対する学習精度をcsvファイルに作成

train_accuracy_csv = basic_name + '_train_validation_accuracy_loss.csv'
f = open(train_accuracy_csv,'a')
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


## 評価用データを使って予測精度を出力させる
## 以下では正解labelとモデルの予測モデルをtextファイルとして出力する

output_file = basic_name + '_label_output.dat' 
comfirm_file = basic_name + '_comfirm_training.dat'
output_Results(val_directory,numofvalidation_data,output_file) #評価用データに対する結果を保存
output_Results(train_directory,numoftraining_data,comfirm_file)#学習に用いたデータに対しても保存

print("### Complete ###")
