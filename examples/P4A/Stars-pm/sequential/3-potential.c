#include <math.h>
#include <fftw3.h>

#include "stars-pm.h"


static void int2float(int v1[NP][NP][NP], float v2[NP][NP][NP]) {
  int i, j, k;
#ifdef P4A_CUDA_CHEAT               // 0.17ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 4.1ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        v2[i][j][k] = v1[i][j][k] - 1; // Avg density = 0
      }
    }
  }
}
/*
static void float2int(float v1[NP][NP][NP], int v2[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        v2[i][j][k] = v1[i][j][k];
      }
    }
  }
}
*/
static void real2Complex(float cdens[NP][NP][NP][2],
                  float dens[NP][NP][NP]) {
  int i, j, k;
#ifdef P4A_CUDA_CHEAT               // 0.31ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 4.7ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        cdens[i][j][k][0] = dens[i][j][k];
        cdens[i][j][k][1] = 0;
      }
    }
  }
}
/*
static void complex2Real(float cdens[NP][NP][NP][2],
                         float dens[NP][NP][NP]) {

  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        dens[i][j][k] = cdens[i][j][k][0];
      }
    }
  }
}
*/
static void complex2Real_correctionPot(float cdens[NP][NP][NP][2],
                                       float dens[NP][NP][NP],
                                       float coeff) {
  int i, j, k;
#ifdef P4A_CUDA_CHEAT               // 0.25ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 4.1ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        dens[i][j][k] = (float)(cdens[i][j][k][0]) * coeff / (DX * DX * DX);
      }
    }
  }
}
/*
static void correctionPot(float pot[NP][NP][NP],
                          float coeff) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        pot[i][j][k] = (float)(pot[i][j][k]) * coeff / (DX * DX * DX);
      }
    }
  }
}
*/
static void fft_laplacian7(float field[NP][NP][NP][2]) {
  int i, j, k;
  float i2, j2, k2;
  int limit = NP >> 2;
  float coeff = M_PI / NP;

#ifdef P4A_CUDA_CHEAT               // 1.02ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 5.2ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        int offset;
        float coeff2;
        if(i > limit) {
          offset = NP;
        } else {
          offset = 0;
        }
        i2 = sinf(coeff * (i - offset));

        offset = 0;
        if(j > limit) {
          offset = NP;
        } else {
          offset = 0;
        }
        j2 = sinf(coeff * (j - offset));

        if(k > limit) {
          offset = NP;
        } else {
          offset = 0;
        }
        k2 = sinf(coeff * (k - offset));

        k2 = k2 * k2 + i2 * i2 + j2 * j2;
        k2 += (k2 == 0);

        coeff2 = G * M_PI * DX * DX / k2 / NP / NP / NP;
        field[i][j][k][0] = field[i][j][k][0] * coeff2;
        field[i][j][k][1] = field[i][j][k][1] * coeff2; // FFT NORMALISATION
/*
        if(i==0&&j==0&&k==0) {
          field[0][0][0][0] = 0;
          field[0][0][0][1] = 0;
        }*/
      }
    }
  }


}


static fftwf_plan fft_forward;
static fftwf_plan fft_backward;

void potential_init_plan(float cdens[NP][NP][NP][2]) {
#ifdef FFTW3_THREADED
  int nthreads = omp_get_max_threads();
  if(nthreads>8) nthreads=8;
  fftwf_init_threads();
#ifndef P4A_BENCH
  fprintf(stderr,"Initialising threaded FFTW3 with %d threads\n",nthreads);
#endif
  fftwf_plan_with_nthreads(nthreads);
#endif
  fft_forward = fftwf_plan_dft_3d(NP,
                                  NP,
                                  NP,
                                  (fftwf_complex*)cdens,
                                  (fftwf_complex*)cdens,
                                  FFTW_FORWARD,
                                  FFTW_ESTIMATE);
  fft_backward = fftwf_plan_dft_3d(NP,
                                   NP,
                                   NP,
                                   (fftwf_complex*)cdens,
                                   (fftwf_complex*)cdens,
                                   FFTW_BACKWARD,
                                   FFTW_ESTIMATE);
}

void potential_free_plan() {
  fftwf_destroy_plan(fft_forward);
  fftwf_destroy_plan(fft_backward);
}

void potential(int histo[NP][NP][NP],
               float dens[NP][NP][NP],
               float cdens[NP][NP][NP][2],
               float mp[NP][NP][NP] ) {

  // Conversion
  TIMING(int2float(histo,dens));

  //************************************ Laplace Solver  ************************************

  TIMING(real2Complex(cdens,dens)); /* conversion de format*/
  TIMING(fftwf_execute(fft_forward)); /* repeat as needed */
  TIMING(fft_laplacian7(cdens));
  TIMING(fftwf_execute(fft_backward)); /* repeat as needed */


  TIMING(complex2Real_correctionPot(cdens,dens,mp[0][0][0]));

/*  complex2Real(cdens, dens);
  correctionPot(dens, mp[0][0][0]);*/
}

