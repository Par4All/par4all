/**
 * LICENSE TERMS

Copyright (c)2008-2010 University of Virginia
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted without royalty fees or other restrictions, provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Virginia, the Dept. of Computer Science, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you use this software or a modified version of it, please cite the most relevant among the following papers:

1) S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, Sang-Ha Lee and K. Skadron.
"Rodinia: A Benchmark Suite for Heterogeneous Computing". IEEE International Symposium
on Workload Characterization, Oct 2009.

2) J. Meng and K. Skadron. "Performance Modeling and Automatic Ghost Zone Optimization
for Iterative Stencil Loops on GPUs." In Proceedings of the 23rd Annual ACM International
Conference on Supercomputing (ICS), June 2009.

3) L.G. Szafaryn, K. Skadron and J. Saucerman. "Experiences Accelerating MATLAB Systems
Biology Applications." in Workshop on Biomedicine in Computing (BiC) at the International
Symposium on Computer Architecture (ISCA), June 2009.

4) M. Boyer, D. Tarjan, S. T. Acton, and K. Skadron. "Accelerating Leukocyte Tracking using CUDA:
A Case Study in Leveraging Manycore Coprocessors." In Proceedings of the International Parallel
and Distributed Processing Symposium (IPDPS), May 2009.

5) S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, and K. Skadron. "A Performance
Study of General Purpose Applications on Graphics Processors using CUDA" Journal of
Parallel and Distributed Computing, Elsevier, June 2008.

6) S. Che, J. Li, J. W. Sheaffer, K. Skadron, and J. Lach. "Accelerating Compute
Intensive Applications with GPUs and FPGAs" In Proceedings of the IEEE Symposium
on Application Specific Processors (SASP), June 2008.
 *
 */


/**
 * This file was cleaned and adapted by Mehdi Amini
 * 11 august 2011
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <srad.h>

// includes, project
#include <cuda.h>

// includes, kernels
#include <srad_kernel.cu>

#include "timing.h"


void random_matrix(float *I, int rows, int cols);
void runTest(int argc, char** argv);
void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n",
          argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");

  exit(1);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  runTest(argc, argv);

  return EXIT_SUCCESS;
}

void runTest(int argc, char** argv) {
  int rows, cols, size_I, size_R, niter = 10, iter;
  float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

#ifdef CPU
  float Jc, G2, L, num, den, qsqr;
  int *iN,*iS,*jE,*jW, k;
  float *dN,*dS,*dW,*dE;
  float cN,cS,cW,cE,D;
#endif

#ifdef GPU

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

#endif

  unsigned int r1, r2, c1, c2;
  float *c;

  if(argc == 9) {
    rows = atoi(argv[1]); //number of rows in the domain
    cols = atoi(argv[2]); //number of cols in the domain
    if((rows % 16 != 0) || (cols % 16 != 0)) {
      fprintf(stderr, "rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1 = atoi(argv[3]); //y1 position of the speckle
    r2 = atoi(argv[4]); //y2 position of the speckle
    c1 = atoi(argv[5]); //x1 position of the speckle
    c2 = atoi(argv[6]); //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations

  } else {
    usage(argc, argv);
  }

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  I = (float *)malloc(size_I * sizeof(float));
  J = (float *)malloc(size_I * sizeof(float));
  c = (float *)malloc(sizeof(float) * size_I);

#ifdef CPU

  iN = (int *)malloc(sizeof(unsigned int*) * rows);
  iS = (int *)malloc(sizeof(unsigned int*) * rows);
  jW = (int *)malloc(sizeof(unsigned int*) * cols);
  jE = (int *)malloc(sizeof(unsigned int*) * cols);

  dN = (float *)malloc(sizeof(float)* size_I);
  dS = (float *)malloc(sizeof(float)* size_I);
  dW = (float *)malloc(sizeof(float)* size_I);
  dE = (float *)malloc(sizeof(float)* size_I);

  for (int i=0; i< rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }
  for (int j=0; j< cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }
  iN[0] = 0;
  iS[rows-1] = rows-1;
  jW[0] = 0;
  jE[cols-1] = cols-1;

#endif

#ifdef GPU

  //Allocate device memory
  cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
  cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
  cudaMalloc((void**)& E_C, sizeof(float)* size_I);
  cudaMalloc((void**)& W_C, sizeof(float)* size_I);
  cudaMalloc((void**)& S_C, sizeof(float)* size_I);
  cudaMalloc((void**)& N_C, sizeof(float)* size_I);

#endif

  //printf("Randomizing the input matrix\n");
  //Generate a random matrix
  random_matrix(I, rows, cols);

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }

  //	printf("Start the SRAD main loop\n");

  /* Start timer. */
  timer_start();

  //Copy data from main memory to device memory
  // Here is an optimization from Mehdi Amini, the original version exposed the
  // transfer inside the loop
  cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    //Currently the input size must be divided by 16 - the block size
    int block_x = cols/BLOCK_SIZE;
    int block_y = rows/BLOCK_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x , block_y);

    //Run kernels
    srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr);
    srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr);

    //Copy data from device memory to main memory
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);

  }

  cudaThreadSynchronize();


  /* Stop timer and display. */
  timer_stop_display();

#ifdef OUTPUT
  //Printing output
  printf("Printing Output:\n");
  for( int i = 0; i < rows; i++) {
    for ( int j = 0; j < cols; j++) {
      printf("%.5f ", J[i * cols + j]);
    }
    printf("\n");
  }
#endif 

  //printf("Computation Done\n");

  free(I);
  free(J);
#ifdef CPU
  free(iN); free(iS); free(jW); free(jE);
  free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
  cudaFree(C_cuda);
  cudaFree(J_cuda);
  cudaFree(E_C);
  cudaFree(W_C);
  cudaFree(N_C);
  cudaFree(S_C);
#endif 
  free(c);
  
}

void random_matrix(float *I, int rows, int cols) {

  srand(7);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }

}

