/*-------- conv-cpu.cpp ---------------*- C++ -*-
 *
 *  (c) HPC Project  -  2009
 *
 */



#include <stdlib.h>
#include <stdbool.h>


/*
 *  a00 * a[idx-n-1]     a10 * a[idx-1]      a20 * a[idx+n-1]
 *  a01 * a[idx-n]       a11 * a[idx]        a21 * a[idx+n]
 *  a02 * a[idx-n+1]     a12 * a[idx+1]      a22 * a[idx+n+1]
 */

void conv_cpu(float *a, float *c, int n, \
	      float a00, float a10, float a20, \
	      float a01, float a11, float a21,  \
	      float a02, float a12, float a22)
{
  int i, j;
  for (i=0; i<n; ++i) {
    for (j=0; j<n; ++j) {
      int idx = i * n + j;

      bool right = i > 0;
      bool left = i < n-1;
      bool top = j > 0;
      bool bottom = j < n-1;

      c[idx] = ((right & top) ? a00 * a[idx-n-1] : 0)
	+ (right ? a10 * a[idx-1] : 0)
	+ ((right & bottom) ? a20 * a[idx+n-1] : 0)
	+ (top ? a01 * a[idx-n] : 0)
	+ a11 * a[idx]
	+ (bottom ? a21 * a[idx+n] : 0)
	+ ((left & top) ? a02 * a[idx-n+1] : 0)
	+ (left ? a12 * a[idx+1] : 0)
	+ ((left & bottom) ? a22 * a[idx+n+1] : 0);
    }
  }
}




//------------------------------------------------------------------------------
// Main program
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int n = 10000;
  int loops = 10;

  int numBytes = n * n * sizeof(float);
  float* h_A = (float *) malloc(numBytes);
  float* h_C = (float *) malloc(numBytes);
  int i, j;
  for(i=0; i< n; ++i) {
    for(j=0; j<n; ++j) {
      h_A[i*n+j] = 1;
      h_C[i*n+j] = 0;
    }
  }


  for (i=0; i<loops; ++i) {
    conv_cpu(h_A, h_C, n, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  }


}
