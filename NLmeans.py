# The following code is based on the original Matlab code by Jose Manjon 
# Herrera & Antoni Buades (09-03-2006), which is the implementation of 
# the Non local filter proposed for A. Buades, B. Coll and J.M. Morel in
# "A non-local algorithm for image denoising". The code is then translated to
# python and GPU-accelerated using numba jit.
# 
# https://www.mathworks.com/matlabcentral/fileexchange/13176-non-local-means-filter
# ------------------------------------------------------------------------
# AUTHOR:   Krittanon "Pond" Sirorattanakul
#           Seismological Laboratory
#           California Institute of Technology
#           Email: krittanon.pond@gmail.com
# ------------------------------------------------------------------------
# REVISION: 1.0.0       02 JUN 2021     Initial creation
# ------------------------------------------------------------------------


import numpy as np
import numpy.matlib
from numba import jit,cuda

##### ---------------------------- CPU Code (Naive python) ---------------------------- #####

def NLmeans(input,f,t,h):
    # Get the image size
    m = input.shape[0]
    n = input.shape[1]

    # Initialize the output matrix
    output = np.zeros((m,n))
    
    # Pad the input for efficient computations
    input_pad = np.pad(input,(f,f),'symmetric')
    
    # Make Gaussian kernel
    kernel = make_kernel(f) 
    kernel = kernel / sum(sum(kernel))

    h2 = h*h
    for i in range(1,m+1):
        for j in range(1,n+1):
            # window center in padded image
            i1 = i + f
            j1 = j + f

            W1 = input_pad[i1-f-1:i1+f,j1-f-1:j1+f]

            wmax = 0
            average = 0
            sweight = 0

            # limits of image search area
            rmin = max(i1-t,f+1)
            rmax = min(i1+t,m+f)
            smin = max(j1-t,f+1)
            smax = min(j1+t,n+f)

            for r in range(rmin,rmax+1):
                for s in range(smin,smax+1):
                    if (r==i1 and s==j1):
                        continue  # Go to next for loop iteration

                    W2 = input_pad[r-f-1:r+f,s-f-1:s+f]
                    d = sum(sum( np.multiply(kernel, np.multiply((W1-W2),(W1-W2)) ) ))                                         
                    w = np.exp(-d/h2)                
                                    
                    if w > wmax:                
                        wmax=w
                    
                    sweight = sweight + w
                    average = average + w * input_pad[r-1,s-1]                                

            average = average + wmax*input_pad[i1-1,j1-1]
            sweight = sweight + wmax
                      
            if sweight > 0:
                output[i-1,j-1] = average / sweight;
            else:
                output[i-1,j-1] = input[i-1,j-1];
                print(i,j)
    return output


# Make Gaussian kernel
def make_kernel(f):
    kernel = np.zeros((2*f+1,2*f+1))
    for d in range(1,f+1):
        value = 1 / (2*d+1)**2
        for i in range(-d,d+1):
            for j in range(-d,d+1):
                kernel[f-i,f-j] += value
    kernel = kernel / f
    return kernel
   

    
    
##### ---------------------------- CPU Code (Numba jit) ---------------------------- #####

@jit
def NLmeans_jit(input,f,t,h):

    # Get the image size
    dim = np.shape(input)
    m = dim[0]
    n = dim[1]

    # Initialize the output matrix
    output = np.zeros((m,n))
    
    # Pad the input for efficient computations
    input_pad = np.pad(input,(f,f),'symmetric')
    
    # Make Gaussian kernel
    kernel = make_kernel_jit(f) 
    kernel = kernel / kernel.sum()

    h2 = h*h

    for i in range(1,m+1):
        for j in range(1,n+1):
            # window center in padded image
            i1 = i + f
            j1 = j + f

            W1 = input_pad[i1-f-1:i1+f,j1-f-1:j1+f]

            wmax = 0
            average = 0
            sweight = 0

            # limits of image search area
            rmin = max(i1-t,f+1)
            rmax = min(i1+t,m+f)
            smin = max(j1-t,f+1)
            smax = min(j1+t,n+f)

            for r in range(rmin,rmax+1):
                for s in range(smin,smax+1):
                    if (r==i1 and s==j1):
                        continue  # Go to next for loop iteration

                    W2 = input_pad[r-f-1:r+f,s-f-1:s+f]
                    d = np.multiply(kernel, np.multiply((W1-W2),(W1-W2)))                                    
                    w = np.exp(-d.sum()/h2)                
                                    
                    if w > wmax:                
                        wmax=w
                    
                    sweight = sweight + w
                    average = average + w * input_pad[r-1,s-1]                                

            average = average + wmax*input_pad[i1-1,j1-1]
            sweight = sweight + wmax
                      
            if sweight > 0:
                output[i-1,j-1] = average / sweight;
            else:
                output[i-1,j-1] = input[i-1,j-1];
                print(i,j)
    return output
    
@jit
def make_kernel_jit(f):
    kernel = np.zeros((2*f+1,2*f+1))
    for d in range(1,f+1):
        value = 1 / (2*d+1)**2
        for i in range(-d,d+1):
            for j in range(-d,d+1):
                kernel[f-i,f-j] += value
    kernel = kernel / f
    return kernel
    

    
    
##### ---------------------------- GPU Code (CUDA) ---------------------------- #####   
@cuda.jit
def NLmeans_GPU_kernel(input_pad,f,t,h,m,n,kernel,output): 
    i, j = cuda.grid(2)
    
    # window center in padded image
    i1 = i + f + 1
    j1 = j + f + 1

    wmax = 0
    average = 0
    sweight = 0

    # limits of image search area
    rmin = max(i1-t,f+1)
    rmax = min(i1+t,m+f)
    smin = max(j1-t,f+1)
    smax = min(j1+t,n+f)

    h2 = h*h

    for r in range(rmin,rmax+1):
        for s in range(smin,smax+1):
            if (r==i1 and s==j1):
                continue  # Go to next for loop iteration
                
            d = 0
            for p in range(len(kernel)):
                for q in range(len(kernel[0])):
                    w_diff = input_pad[i1-f-1+p,j1-f-1+q] - input_pad[r-f-1+p,s-f-1+q]
                    d = d + kernel[p,q] * w_diff * w_diff
            w = 2.718281828459**(-d/h2)                
                                    
            if w > wmax:                
                wmax=w
                    
            sweight = sweight + w
            average = average + w * input_pad[r-1,s-1]                                

    average = average + wmax*input_pad[i1-1,j1-1]
    sweight = sweight + wmax
                      
    output[i,j] = average / sweight;



@jit
def NLmeans_GPU(input,f,t,h,blocks,threads_per_block):

    # Get the image size
    dim = np.shape(input)
    m = dim[0]
    n = dim[1]

    # Initialize the output matrix
    output = np.zeros((m,n))
    input_pad = np.pad(input,(f,f),'symmetric')
    kernel = make_kernel_jit(f) 
    kernel = kernel / kernel.sum()

    NLmeans_GPU_kernel[blocks, threads_per_block](input_pad,f,t,h,m,n,kernel,output)

    return output
