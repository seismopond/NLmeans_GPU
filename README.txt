------------------------------------------------------------------------
GPU-ACCELERATED NON-LOCAL MEANS FILTER
------------------------------------------------------------------------
GENERAL DESCRIPTIONS

This is a library for GPU-accelerated non-local means filter. This data 
filtering technique is generally used for displacement fields derived
from digital image correlation (DIC). In contrary to the local filter, 
in which all neighboring pixels are used to smooth the displacement 
fields regardless of their content, this non-local means filter weights
the neighboring pixels by the degree of similiarity. As a result, this
non-local means filtering will preserve sharp edges and large gradients.

There are a few parameters that you have to choose when running the 
filters. These include:
	f: radius of similarity window
	t: radius of search window
	h: half-amplitude of the noise level

The original implementation in which this code is based on was written
by Jose Manjon Herrera & Antoni Buades (09-03-2006) in Matlab. The 
algorithm is based on A. Buades, B. Coll and J.M. Morel in
"A non-local algorithm for image denoising". The code is then translated
to python and GPU-accelerated using numba jit and CUDA kernels.

The original Matlab implementation can be found here
https://www.mathworks.com/matlabcentral/fileexchange/13176-non-local-means-filter

------------------------------------------------------------------------
REQUIRED LIBRARIES
	numpy
	scipy
	matplotlib
	numba

------------------------------------------------------------------------
DEMO CODE

The jupyter notebook "NLmeans_demo.ipynb" demonstrates how the library
can be run through Google Colab. The demo code uses one of the two 
dataset included as a test data.

------------------------------------------------------------------------
TEST DATA

The dataset provided is in .mat file from Matlab and can be read in by 
using scipy.io function loadmat. 

The first dataset is "NLmeans_test_small.mat". The data is 30 pixels by
30 pixels. "FN" is the raw data. "FN_fil" is the filtered version of 
"FN" using f=3, t=21, h=0.5 and the original Matlab code.

The second dataset is "NLmeans_test_data.mat". The data is 100 pixels by
100 pixels. "FN" is the raw data. "FN_fil" is the filtered version of 
"FN" using f=3, t=41, h=0.5.

Both dataset are obtained from the real experiments conducted in Rosakis
Lab at California Institute of Technology.

------------------------------------------------------------------------
PERFORMANCES

The first dataset is relatively small and hence the GPU performances is
not properly justified due to significant overhead for copying the data
between the host CPU and the device GPU. The following results are the
performances using the second dataset.

- Original Matlab code (Intel Core i7-10510U CPU): 91.03 seconds
- Python native (Intel Xeon 2.30 GHz CPU): 691.89 seconds
- Python numba jit (Intel Xeon 2.30 GHz CPU): 14.18 seconds
- Numba CUDA kernels (NVIDIA Teslat T4): 0.20 seconds

The GPU-accelerated code give over 3,500 tims acceleration over python
native and 70 times over numba jit.

------------------------------------------------------------------------
 AUTHOR:   Krittanon "Pond" Sirorattanakul
           Seismological Laboratory
           California Institute of Technology
           Email: krittanon.pond@gmail.com
------------------------------------------------------------------------
 REVISION: 1.0.0       02 JUN 2021     Initial creation
------------------------------------------------------------------------