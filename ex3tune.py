import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
import numpy as np
from keras.utils import np_utils
from keras import backend as K
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#hyperparameters
nh1=[64,128]
nh2 = [256, 512]
a = [0.1, 0.001, 0.000001]
lr = [0.1, 0.01, 0.001]

#code for F1-measure from: https://datascience.stackexchange.com/a/45166
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#hypermodel
initializer = tf.keras.initializers.HeNormal()
def build_model(hp):
  reg=hp.Choice('reg',a)
  model = keras.Sequential()
  model.add( keras.layers.Flatten(input_shape=(28,28)))
  model.add(keras.layers.Dense(
      hp.Choice('units1', nh1),
      activation='relu',
      kernel_initializer=initializer,
      kernel_regularizer=regularizers.l2(reg)))
  model.add(keras.layers.Dense(
      hp.Choice('units2', nh2),
      activation='relu',
      kernel_initializer=initializer,
      kernel_regularizer=regularizers.l2(reg)))
  model.add(layers.Dense(10,activation='softmax'))
  learning_rate = hp.Choice('learning_rate', lr)
 
  
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
      loss=tf.losses.CategoricalCrossentropy(),
      metrics=['accuracy',f1_m,precision_m, recall_m])
  return model
    

tuner = kt.Hyperband(build_model,objective=kt.Objective("val_f1_m", direction="max"),max_epochs=1000,directory='No1')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=200, mode="max")

tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[callback])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)
eval_result = model.evaluate(x_test, y_test);
pred = model.predict(x_test)
con_matrix = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
ax = sns.heatmap(con_matrix, annot=True, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix\n')
ax.set_xlabel('\nPredicted')
ax.set_ylabel('Actual')
plt.show()

# summarize history for F1_measure
plt.plot(history.history['f1_m'], label='train')
plt.plot(history.history['val_f1_m'], label='validation')
plt.title('Model F1_Measure')
plt.xlabel('Epochs')
plt.ylabel('F1_Measure')
plt.legend()
plt.show()
# summarize history for accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

print('accuracy:',eval_result[1])
print('recall:',eval_result[4])
print('precision:',eval_result[3])
print('F1_Measure:',eval_result[2])

print('Optimal node amount for first hidden layer:',best_hps.get('units1'))
print('Optimal node amount for second hidden layer:',best_hps.get('units2'))
print('Optimal learning rate:',best_hps.get('learning_rate'))
print('Optimal L2 parameter:',best_hps.get('reg'))






