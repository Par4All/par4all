#include <math.h>
#include <fftw3.h>

#include "stars-pm.h"


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

static void fft_laplacian7(float field[NP][NP][NP][2]) {
  int i, j, k;
  float i2, j2, k2;

  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        i2 = (i > NP / 2. ? sinf(M_PI / NP * (i - NP)) * sinf(M_PI
            / NP * (i - NP)) : sinf(M_PI / NP * i) * sinf(M_PI / NP
            * i));
        j2 = (j > NP / 2. ? sinf(M_PI / NP * (j - NP)) * sinf(M_PI
            / NP * (j - NP)) : sinf(M_PI / NP * j) * sinf(M_PI / NP
            * j));
        k2 = (k > NP / 2. ? sinf(M_PI / NP * (k - NP)) * sinf(M_PI
            / NP * (k - NP)) : sinf(M_PI / NP * k) * sinf(M_PI / NP
            * k)) + i2 + j2;
        k2 += (k2 == 0);

        field[i][j][k][0] = field[i][j][k][0] * G * M_PI * DX * DX / k2 / NP
            / NP / NP;
        field[i][j][k][1] = field[i][j][k][1] * G * M_PI * DX * DX / k2 / NP
            / NP / NP; // FFT NORMALISATION

      }
    }
  }

  field[0][0][0][0] = 0;
  field[0][0][0][1] = 0;

}


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

static fftwf_plan fft_forward;
static fftwf_plan fft_backward;

void potential_init_plan(float cdens[NP][NP][NP][2]) {
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
  int2float(histo, dens);

  //************************************ Laplace Solver  ************************************

  real2Complex(cdens, dens); /* conversion de format*/
  fftwf_execute(fft_forward); /* repeat as needed */
  fft_laplacian7(cdens);
  fftwf_execute(fft_backward); /* repeat as needed */
  complex2Real(cdens, dens); // conversion de format


  correctionPot(dens, mp[0][0][0]);

}

