import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/home/furkan/İndirilenler/train.csv.zip')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


# Bu kısma kadar kopyala yapıştır yaptım ama
# bundan sonraki forward prop kısmını kendim ve oop kullanarak yazdım


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights=0.10*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.outputs=np.dot(inputs,self.weights)+self.bias


class Activation_ReLu:
    def forward(self,inputs):
        self.outputs=np.maximum(inputs,0)


class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probilities=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=probilities

dense1=Layer_Dense(41000,10)
activation1=Activation_ReLu()

dense2=Layer_Dense(10,10)
activation2=Activation_Softmax()

dense1.forward(Y_train)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)
print(activation2.output[:5])
