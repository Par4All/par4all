#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


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
    // There shouldn't be a copy-in there because J is not written but only read in the previous loop
    // This is a currently known limitation of the communication optimization scheme, hard work to improve !
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


