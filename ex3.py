import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #normalize in (0,1) interval

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train, x_val = tf.split(x_train, [48000, 12000], 0)
y_train, y_val = tf.split(y_train, [48000, 12000], 0)


RMS1 = tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.01)
RMS2 = tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.99)
SGDopt = tf.keras.optimizers.SGD(learning_rate=0.01)
initializer = tf.keras.initializers.RandomNormal(mean=0.)


model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model1.compile(optimizer=RMS1, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])

model2.compile(optimizer=RMS2, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_initializer=initializer),
    keras.layers.Dense(units=128, activation='relu', kernel_initializer=initializer),
    keras.layers.Dense(units=10, activation='softmax')
])

model3.compile(optimizer=SGDopt, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model1_L2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    keras.layers.Dense(units=10, activation='softmax')
])

model1_L2.compile(optimizer=RMS1, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model2_L2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(units=10, activation='softmax')
])

model2_L2.compile(optimizer=RMS2, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model3_L2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(units=128, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(units=10, activation='softmax')
])

model3_L2.compile(optimizer=SGDopt, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model1_L1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(units=10, activation='softmax')
])

model1_L1.compile(optimizer=RMS1, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model2_L1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=10, activation='softmax')
])

model2_L1.compile(optimizer=RMS2, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


model3_L1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(units=256, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(units=128, activation='relu', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(units=10, activation='softmax')
])

model3_L1.compile(optimizer=SGDopt, 
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])



history1 = model1.fit(  
    x_train,y_train,
    batch_size=256,
    epochs=100, 
    validation_data=(x_val, y_val), 
    )

history2 = model2.fit(  
    x_train,y_train,
    batch_size=256,
    epochs=100, 
    validation_data=(x_val, y_val), 
    )

history3 = model3.fit(  
    x_train,y_train,
    batch_size=256,
    epochs=100, 
    validation_data=(x_val, y_val), 
    )

history1_L2 = model1_L2.fit(  
    x_train,y_train,
    batch_size=256,
    epochs=100, 
    validation_data=(x_val, y_val), 
    )

history2_L2 = model2_L2.fit(
    x_train,y_train,
    epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val), 
    )

history3_L2 = model3_L2.fit(  
    x_train,y_train, 
    epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val), 
    )

history1_L1 = model1_L2.fit(  
    x_train,y_train, 
    epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val), 
    )

history2_L1 = model2_L2.fit(
    x_train,y_train,
    epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val), 
    )

history3_L1 = model3_L2.fit(  
    x_train,y_train, 
    epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val), 
    )

# summarize history for accuracy
plt.plot(history1.history['accuracy'], label='train')
plt.plot(history1.history['val_accuracy'], label='validation')
plt.title('Model1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='validation')
plt.title('Model1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


# summarize history for accuracy
plt.plot(history2.history['accuracy'], label='train')
plt.plot(history2.history['val_accuracy'], label='validation')
plt.title('Model2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='validation')
plt.title('Model2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history3.history['accuracy'], label='train')
plt.plot(history3.history['val_accuracy'], label='validation')
plt.title('Model3 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history3.history['loss'], label='train')
plt.plot(history3.history['val_loss'], label='validation')
plt.title('Model3 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history1_L2.history['accuracy'], label='train')
plt.plot(history1_L2.history['val_accuracy'], label='validation')
plt.title('Model1_L2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history1_L2.history['loss'], label='train')
plt.plot(history1_L2.history['val_loss'], label='validation')
plt.title('Model1_L2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history2_L2.history['accuracy'], label='train')
plt.plot(history2_L2.history['val_accuracy'], label='validation')
plt.title('Model2_L2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history2_L2.history['loss'], label='train')
plt.plot(history2_L2.history['val_loss'], label='validation')
plt.title('Model2_L2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history3_L2.history['accuracy'], label='train')
plt.plot(history3_L2.history['val_accuracy'], label='validation')
plt.title('Model3_L2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history3_L2.history['loss'], label='train')
plt.plot(history3_L2.history['val_loss'], label='validation')
plt.title('Model3_L2 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


# summarize history for accuracy
plt.plot(history1_L1.history['accuracy'], label='train')
plt.plot(history1_L1.history['val_accuracy'], label='validation')
plt.title('Model1_L1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history1_L1.history['loss'], label='train')
plt.plot(history1_L1.history['val_loss'], label='validation')
plt.title('Model1_L1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history2_L1.history['accuracy'], label='train')
plt.plot(history2_L1.history['val_accuracy'], label='validation')
plt.title('Model2_L1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history2_L1.history['loss'], label='train')
plt.plot(history2_L1.history['val_loss'], label='validation')
plt.title('Model2_L1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()



# summarize history for accuracy
plt.plot(history3_L1.history['accuracy'], label='train')
plt.plot(history3_L1.history['val_accuracy'], label='validation')
plt.title('Model3_L1 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history3_L1.history['loss'], label='train')
plt.plot(history3_L1.history['val_loss'], label='validation')
plt.title('Model3_L1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

eval_result1 = model1.evaluate(x_test, y_test);
eval_result2 = model2.evaluate(x_test, y_test);
eval_result3 = model3.evaluate(x_test, y_test);
eval_result1_L2 = model1_L2.evaluate(x_test, y_test);
eval_result2_L2 = model2_L2.evaluate(x_test, y_test);
eval_result3_L2 = model3_L2.evaluate(x_test, y_test);
eval_result1_L1 = model1_L1.evaluate(x_test, y_test);
eval_result2_L1 = model2_L1.evaluate(x_test, y_test);
eval_result3_L1 = model3_L1.evaluate(x_test, y_test);

