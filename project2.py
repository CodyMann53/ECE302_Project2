#read image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

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
    output = np.zeros( (M-8, N-8))

    #loop through all of the rows of the image
    for i in range(M - 8):

        #loop through all of the cols of image
        for j in range(N-8):

            #extract an 8x8 patch of pixels
            z = Y[i:i+8, j:j+8]

            #turn 8x8 array into 64x1 col vector
            z_vector = z.flatten('F')

            z_vector = z_vector.reshape((64,1))

            #if ( f(cat|z) > f(grass|z), then set output(i,j) = 1
            if(Gcat(mu_cat, Sigma_cat, z_vector, Acat, Bcat, Ccat) >
            Ggrass(mu_grass, Sigma_grass, z_vector, Agrass, Bgrass, Cgrass )):

                output[i,j] = 1

            else:

                #else set output(i,j) = 0
                output[i,j] = 0

    return output

def pad(z_vector):

    #while the vector size does not have 64 elements
    while (z_vector.size < 64):

        #append 0 because most likely was at an edge on cat pixel will not be there
        z_vector = np.append(z_vector, 0)

    #return the new padded vector
    return z_vector

def Gcat(mu_cat, Sigma_cat, z, Acat, Bcat, Ccat):

    return ( Acat - Bcat - (0.5) * np.matmul(np.matmul( np.transpose( z- mu_cat ), Ccat ), (z - mu_cat ) ) )

def Ggrass(mu_grass, Sigma_grass, z, Agrass, Bgrass, Cgrass ):

    return ( Agrass - Bgrass - (0.5) * np.matmul(np.matmul( np.transpose( z- mu_grass ), Cgrass ), (z - mu_grass ) ) )

def f_cat(K_cat, K_grass):

    return ( K_cat / (K_cat + K_grass))

def f_grass(K_cat, K_grass):

    return ( K_grass / (K_cat + K_grass))

#def MAE(X, X_truth):

    #result = np.abs(np.subtract(X, X_truth))

##MAIN
train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','))

# 1 calculate the sample mean and sample covariance for the training sets
mu_cat, mu_grass, Sigma_cat, Sigma_grass = my_training(train_cat, train_grass)

#read in image and divide all points by 255 to normalize values between (0,1)
Y = ( plt.imread('cat_grass.jpg') / 255 )

start_time = time.time()
# 2. process image
output = my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass, M, N)
print('My runtime is %s seconds' % (time.time() - start_time))

print(output.shape)

# 4. plot processed image
plt.imshow(output * 255, cmap='gray')
plt.show()
