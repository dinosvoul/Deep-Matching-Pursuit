
from DeepMP import deepmp
import numpy as np
from keras.layers import Input
from sklearn.model_selection import train_test_split
from numpy import random
from numpy import array
from keras_adabound import AdaBound

adab=AdaBound(lr=1e-3,final_lr=0.1)



epochs = 20
batch_size = 50
num_samples = 50000
n = 200
m = 30

x = np.zeros((num_samples,n))
y = np.zeros((num_samples,m))
check=np.zeros(num_samples)
x_supp = np.zeros(x.shape)
k=2 #sparsity of the signal


M = array(np.abs(random.standard_normal((m, n))), dtype='float32') 

input_shape = y.shape[1:]
inputs = Input(shape=input_shape)

#Normalize the dictionary as implied by the standard procedure for Matching Pursuit Algorithms. 
for ii in range(0, M.shape[1]):
    mind = M[:, ii]**2
    no = mind.sum()
    M[:, ii] = M[:, ii] / np.sqrt(no)
for ii in range(num_samples):
    p = random.permutation(n)
    x[ii, p[0:k]] = random.uniform(0.2, 1, (k,))
    y[ii, :] = np.dot(M, x[ii, :])



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=30)


model = deepmp(input_shape=input_shape,SenMat=M,k=k)


model.compile(
            loss='categorical_crossentropy',
            optimizer=adab,
            metrics=['accuracy'])

history=model.fit(y_train, x_train, batch_size=batch_size,validation_data=(y_test, x_test), epochs=epochs, verbose=1)
