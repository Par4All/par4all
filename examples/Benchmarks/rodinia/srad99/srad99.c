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
 * This file was converted into C99 form by Mehdi Amini
 * 11 august 2011
 */

// srad.cpp : Defines the entry point for the console application.
//

//#define OUTPUT


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "timing.h"

void random_matrix(int rows, int cols, float I[rows][cols]);

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <no. of threads><lamda> <no. of iter>\n",
          argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1>      - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");

  exit(1);
}

void init(int rows,
          int cols,
          float I[rows][cols],
          float J[rows][cols],
          int iN[rows],
          int iS[rows],
          int jW[cols],
          int jE[cols]) {
  for (int i = 0; i < rows; i++) {
    iN[i] = i - 1;
    iS[i] = i + 1;
  }
  for (int j = 0; j < cols; j++) {
    jW[j] = j - 1;
    jE[j] = j + 1;
  }
  iN[0] = 0;
  iS[rows - 1] = rows - 1;
  jW[0] = 0;
  jE[cols - 1] = cols - 1;

  //printf("Randomizing the input matrix\n");

  random_matrix(rows, cols, I);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      J[i][j] = (float)exp(I[i][j]);
    }
  }

}

int main(int argc, char* argv[]) {
  int rows, cols, size_I, size_R, niter = 10, iter;
  float q0sqr, sum, sum2, tmp, meanROI, varROI;
  float Jc, G2, L, num, den, qsqr;
  int r1, r2, c1, c2;
  float cN, cS, cW, cE;
  float D;
  float lambda;
  int i, j;

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

  float I[rows][cols];
  float J[rows][cols];
  float c[rows][cols];

  int iN[rows];
  int iS[rows];
  int jW[cols];
  int jE[cols];

  float dN[rows][cols];
  float dS[rows][cols];
  float dW[rows][cols];
  float dE[rows][cols];


  // Initial data
  init(rows,cols,I,J,iN,iS,jW,jE);

  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[5]==0) {
    init(rows,cols,I,J,iN,iS,jW,jE);
  }



  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (i = r1; i <= r2; i++) {
      for (j = c1; j <= c2; j++) {
        tmp = J[i][j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {

        Jc = J[i][j];

        // directional derivates
        dN[i][j] = J[iN[i]][j] - Jc;
        dS[i][j] = J[iS[i]][j] - Jc;
        dW[i][j] = J[i][jW[j]] - Jc;
        dE[i][j] = J[i][jE[j]] - Jc;

        G2 = (dN[i][j] * dN[i][j] + dS[i][j] * dS[i][j] + dW[i][j] * dW[i][j]
            + dE[i][j] * dE[i][j]) / (Jc * Jc);

        L = (dN[i][j] + dS[i][j] + dW[i][j] + dE[i][j]) / Jc;

        num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
        den = 1 + (.25 * L);
        qsqr = num / (den * den);

        // diffusion coefficent (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        c[i][j] = 1.0 / (1.0 + den);

        // saturate diffusion coefficent
        if(c[i][j] < 0) {
          c[i][j] = 0;
        } else if(c[i][j] > 1) {
          c[i][j] = 1;
        }
      }

    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {

        // diffusion coefficent
        cN = c[i][j];
        cS = c[iS[i]][j];
        cW = c[i][j];
        cE = c[i][jE[j]];

        // divergence (equ 58)
        D = cN * dN[i][j] + cS * dS[i][j] + cW * dW[i][j] + cE * dE[i][j];

        // image update (equ 61)
        J[i][j] = J[i][j] + 0.25 * lambda * D;
#ifdef OUTPUT
        //printf("%.5f ", J[k]);
#endif //output
      }
#ifdef OUTPUT
      //printf("\n");
#endif //output
    }

  }



  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[5]==0) {
    for( int i = 0; i < rows; i++) {
      for ( int j = 0; j < cols; j++) {
        printf("%.5f ", J[i][j]);
      }
      printf("\n");
    }
  }

  /* Stop timer and display. */
  timer_stop_display();

#ifdef OUTPUT
  for( int i = 0; i < rows; i++) {
    for ( int j = 0; j < cols; j++) {
      printf("%.5f ", J[i][j]);
    }
    printf("\n");
  }
#endif

  //  printf("Computation Done\n");

  return 0;
}

void random_matrix(int rows, int cols, float I[rows][cols]) {

  srand(7);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
#ifndef OUTPUT
      I[i][j] = rand() / (float)RAND_MAX;
#else
      I[i][j] = (float)(i*j)/(float)(rows*cols);
#endif 
    }
#ifdef OUTPUT
    //printf("\n");
#endif
  }

}

