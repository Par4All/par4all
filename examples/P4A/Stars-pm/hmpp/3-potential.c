#include <math.h>
#include <fftw3.h>

#include "stars-pm.h"


#pragma hmpp int2float codelet, target=CUDA
static void int2float(int v1[NP][NP][NP], float v2[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        v2[i][j][k] = v1[i][j][k] - 1; // Avg density = 0
      }
    }
  }
}

#pragma hmpp real2Complex codelet, target=CUDA
static void real2Complex(float cdens[NP][NP][NP][2],
                  float dens[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        cdens[i][j][k][0] = dens[i][j][k];
        cdens[i][j][k][1] = 0;
      }
    }
  }
}

#pragma hmpp complex2Real_correctionPot codelet, target=CUDA
static void complex2Real_correctionPot(float cdens[NP][NP][NP][2],
                                       float dens[NP][NP][NP],
                                       float coeff) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        dens[i][j][k] = (float)(cdens[i][j][k][0]) * coeff / (DX * DX * DX);
      }
    }
  }
}

#pragma hmpp fft_laplacian7 codelet, target=CUDA
static void fft_laplacian7(float field[NP][NP][NP][2]) {
  int i, j, k;
  float i2, j2, k2;
  int limit = NP >> 2;
  float coeff = M_PI / NP;

  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
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
#pragma hmpp int2float callsite
  TIMING(int2float(histo,dens));

  //************************************ Laplace Solver  ************************************

#pragma hmpp real2Complex callsite
  TIMING(real2Complex(cdens,dens)); /* conversion de format*/
  TIMING(fftwf_execute(fft_forward)); /* repeat as needed */
#pragma hmpp fft_laplacian7 callsite
  TIMING(fft_laplacian7(cdens));
  TIMING(fftwf_execute(fft_backward)); /* repeat as needed */


#pragma hmpp complex2Real_correctionPot callsite
  TIMING(complex2Real_correctionPot(cdens,dens,mp[0][0][0]));

/*  complex2Real(cdens, dens);
  correctionPot(dens, mp[0][0][0]);*/
}

