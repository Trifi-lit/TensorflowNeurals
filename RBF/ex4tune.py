import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from keras import backend as Ker
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import Initializer, Constant
from sklearn.cluster import KMeans
from keras.datasets import boston_housing
import keras_tuner as kt

#initialize the centers with Kmeans algorithm
class KMeansCenterInitialize(Initializer):
    def __init__(self, X):
        self.X = X  #training data
        super().__init__()

    def __call__(self, shape, dtype=None, *args): 
        assert shape[1] == self.X.shape[1] #number of data columns(characteristics)
        n_centers = shape[0] #number of rows (observations)
        kmeans = KMeans(n_clusters=n_centers, max_iter=50) #kmeans converges at 20-50 iterations in practical situations 
        kmeans.fit(self.X)
        return kmeans.cluster_centers_
    




class RBFLayer(layers.Layer):
    def __init__(self, output_dim, betas=1.0, initializer=None, **kwargs):
        self.output_dim = output_dim
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=1.0),trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = Ker.expand_dims(self.centers)
        H = Ker.transpose(C-Ker.transpose(x))
        return Ker.exp(-self.betas * Ker.sum(H**2, axis=1))

  
    
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.25)
#the 13 attributes are stored in x and the corresponding prices are stored in y


x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

split_size = int(x_train.shape[0]*0.8)
x_train, x_val = x_train[:split_size], x_train[split_size:]
y_train, y_val = y_train[:split_size], y_train[split_size:]

m = tf.keras.metrics.RootMeanSquaredError()


neurons1=round(0.05*len(x_train))
neurons2=round(0.15*len(x_train))
neurons3=round(0.30*len(x_train))
neurons4=round(0.5*len(x_train))


def build_model(hp):

    RBFn=hp.Choice('RBFlayer',values=[neurons1,neurons2,neurons3,neurons4])
    Outn=hp.Choice('Outputlayer',values=[32,64,128,256])
    Dprob=hp.Choice('dropoutLayer',values=[0.2,0.35,0.5])
    rbflayer_tuned = RBFLayer(RBFn,initializer=KMeansCenterInitialize(x_train),input_shape=(13,))
    
    model=tf.keras.Sequential()
    model.add(rbflayer_tuned)
    model.add(layers.Dense(units=Outn))
    model.add(layers.Dropout(Dprob))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),loss=['mse'],metrics=[m])
    return model

#fit the model 
 
tuner=kt.Hyperband(build_model, objective=kt.Objective("val_loss", direction="min"),max_epochs=100,directory='No2',overwrite=True)

tuner.search(x_train,y_train,epochs=100,validation_split=0.2)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
model.summary()

history=model.fit(x_train,y_train, epochs=300,batch_size=32, validation_split=0.2)
eval_result=model.evaluate(x_test,y_test)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Learning curves of training and validation')
plt.legend()
plt.show()

print('Optimal node amount for hidden(RBF) layer:',best_hps.get('RBFlayer'))
print('Optimal node amount for output layer:',best_hps.get('Outputlayer'))
print('Optimal dropout propability:',best_hps.get('dropoutLayer'))
