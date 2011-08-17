#include <math.h>
#include <stdlib.h>

#define M 1

typedef float FLOAT;
typedef FLOAT RGB[3];

typedef unsigned char IMAGE_RGB[3];


static FLOAT step(FLOAT x) {
#ifdef PIECEWISE_LINEAR
  FLOAT xo = x - 0.5f;
  return
  xo <= (-0.5f / M) ? 0 :
  xo >= (+0.5f / M) ? 1 :
  xo * M;
#else
  return 0.5f * (1.0f + tanhf(M * (x - 0.5f)));
#endif
}


/*
 * Error signal measured between step input and output.  Performance
 * critical.
 */
FLOAT error(FLOAT x) {
  return step(x) - x;
}

/**
 * Compute a uniformly distributed random number between given min and
 * max. Quality of randomness may be poor.
 */
FLOAT uniform_random(FLOAT min, FLOAT max) {
  return min + (max - min) * (rand() / (FLOAT)RAND_MAX);
}



/**
 * For input signal x, do one iteration of error diffusion from buffer
 * ui to buffer uj.  The error convolution kernel is q x q, where q is
 * odd.  The color mixing transform is kInv, a 3 x 3 matrix for R, G,
 * and B components.  Convergence constant alpha, 0 < alpha <= 1 is
 * typically set close to 1/M for good results.  The middle element
 * of the kernel is ignored.  Inner loops are performance critical.
 */
void do_one_iteration(int m,
                      int n,
                      IMAGE_RGB x[m][n],
                      RGB ui[m][n],
                      RGB uj[m][n],
                      int q,
                      FLOAT weights[q][q],
                      FLOAT kInv[3][3],
                      FLOAT alpha) {
  // Offset to take 0..(q-1) to zero-center coordinates,
  // [-r .. r] where r = (q-1)/2
  int qofs = (int)q / -2;

  // Loops over the entire 2d signal
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {

      // Error convolution.
      RGB err = { 0, 0, 0 };
      for (int s = 0; s < q; s++) {
        for (int t = 0; t < q; t++) {
          int i_q = i + qofs + s;
          int j_q = j + qofs + t;

          // Clamp error to 0 when kernel point is outside image.
          if(0 <= i_q && i_q < m && 0 <= j_q && j_q <= n) {
            for (int k = 0; k < 3; k++) {
              err[k] += weights[s][t] * error(ui[i_q][j_q][k]);
            } // for k
          } // if i_q || j_q

        } // for t
      } // for s

      // Error signal for middle of kernel is handled specially.
      RGB err0;
      for (int k = 0; k < 3; k++) {
        err0[k] = err[k] - error(ui[i][j][k]);
      }

      // Color mixing to obtain final error signal.
      for (int s = 0; s < 3; s++) {
        for (int t = 0; t < 3; t++) {
          err[s] += kInv[s][t] * err0[t];
        }
      }

      // Result.
      for (int k = 0; k < 3; k++) {
        uj[i][j][k] = (1.0f - alpha) * ui[i][j][k] + alpha
            * ((FLOAT)(1. / 256.) * x[i][j][k] - err[k]);
      }

    } // for j
  } // for i
}

