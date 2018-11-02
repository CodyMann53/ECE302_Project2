#read image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from NueralNetwork import *

M = 375
N = 500
K_cat = 1976
K_grass = 9556

#takes in two training sets train_cat and train_grass and computes thier sample means and sample covariance
#to be returned
def my_training(train_cat, train_grass):

    mu_cat = np.mean(train_cat, 1)
    mu_grass = np.mean(train_grass, 1)
    Sigma_cat = np.cov(train_cat)
    Sigma_grass = np.cov(train_grass)
    return mu_cat, mu_grass, Sigma_cat, Sigma_grass

#given the sample mean and covariances for training sets cat and grass, takes an image Y and labels data points as either grass or cat
def my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass, M, N):

    #calculate constants
    Acat = np.log( (f_cat(K_cat, K_grass) ) )
    Agrass = np.log( f_grass(K_cat, K_grass) )
    Bcat = (1/2) * np.log( np.linalg.det(Sigma_cat) )
    Bgrass = (1/2) * np.log( np.linalg.det(Sigma_grass) )
    Ccat = np.linalg.pinv(Sigma_cat)
    Cgrass = np.linalg.pinv(Sigma_grass)

    #create a MxN matrix of zeros to be used as the output
    output = np.zeros( (M, N))

    #loop through all of the rows of the image
    for i in range(M-8):

        #loop through all of the cols of image
        for j in range(N-8):

            z = Y[i:i+8, j:j+8]

            #turn 8x8 array into 64x1 col vector
            z_vector = z.flatten('F')

            z_vector = z_vector.reshape((64,1) )

            #if ( f(cat|z) > f(grass|z), then set output(i,j) = 1
            if(Gcat(mu_cat, Sigma_cat, z_vector, Acat, Bcat, Ccat) >
            Ggrass(mu_grass, Sigma_grass, z_vector, Agrass, Bgrass, Cgrass )):

                output[i,j] = 1

            else:

                #else set output(i,j) = 0
                output[i,j] = 0

    return output

def my_testing_new(Y, network):

    #create a MxN matrix of zeros to be used as the output
    output = np.zeros( (M, N))

    #loop through all of the rows of the image
    for i in range(M-8):

        #loop through all of the cols of image
        for j in range(N-8):

            z = Y[i:i+8, j:j+8]

            #turn 8x8 array into 64x1 col vector
            z_vector = z.flatten('F')

            z_vector = z_vector.reshape((64,1) )

            #seed through nueral nework
            networkResult = feedforward(network, z_vector)

            if (networkResult[0] > 0.37) | (networkResult[1] < 0.1):
                output[i,j] = 1
            else:
                output[i,j] = 0

    return output

def Gcat(mu_cat, Sigma_cat, z, Acat, Bcat, Ccat):

    return ( Acat - Bcat - (0.5) * np.matmul(np.matmul( np.transpose( z- mu_cat ), Ccat ), (z - mu_cat ) ) )

def Ggrass(mu_grass, Sigma_grass, z, Agrass, Bgrass, Cgrass ):

    return ( Agrass - Bgrass - (0.5) * np.matmul(np.matmul( np.transpose( z- mu_grass ), Cgrass ), (z - mu_grass ) ) )

def f_cat(K_cat, K_grass):

    return ( K_cat / (K_cat + K_grass))

def f_grass(K_cat, K_grass):

    return ( K_grass / (K_cat + K_grass))

def MAE(X, X_truth):

    diff = np.abs(np.subtract(X,X_truth))
    sum = np.sum(diff)
    sum = sum / X_truth.size
    return sum

##MAIN
train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))

# 1 calculate the sample mean and sample covariance for the training sets
mu_cat, mu_grass, Sigma_cat, Sigma_grass = my_training(train_cat, train_grass)

#read in image and divide all points by 255 to normalize values between (0,1)
Y = ( plt.imread('cat_grass.jpg') / 255)


#create a network
network = Network([64,30,30, 2])

#orgainize the training data into a list of tuples (64x1,2x1)
training_data = createTrainingData(train_cat, train_grass)

#teach network
SGD(network, training_data, 100,10 , 2.0)

output2 = my_testing_new(Y, network)
output3 = my_testing_new(output2, network)

# 2. process image
start_time = time.time()
output = my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, train_cat.size, train_grass.size, M, N)
print('My runtime is %s seconds' % (time.time() - start_time))

#3. MAE
#read in true image
Xstar = plt.imread('truth.png')
mae = MAE(output, Xstar)
mae2 = MAE(output2, Xstar)
mae3 = MAE(output3, Xstar)
print("mae: ", mae*100, "%")
print("mae: ", mae2*100, "%")
print("mae: ", mae3*100, "%")

# plot processed image
plt.imshow(output2 * 255, cmap='gray')
plt.show()
plt.imshow(output * 255, cmap='gray')
plt.show()
plt.imsave('cat_gauss.png', output, cmap='gray')
plt.imsave('network.png', output2, cmap='gray')
plt.imsave('network2.png', output3, cmap='gray')



#come up with better solution. Will try to use nueral network
