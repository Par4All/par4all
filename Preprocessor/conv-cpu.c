/*-------- conv-cpu.cpp ---------------*- C++ -*-
 *
 *  (c) HPC Project  -  2009
 *
 */



#include <stdlib.h>


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

      if (0<i && i<n-1 && 0<j && j<n-1) {
	c[idx] = a00 * a[idx-n-1] + a10 * a[idx-1] + a20 * a[idx+n-1] \
	  + a01 * a[idx-n]   + a11 * a[idx]   + a21 * a[idx+n]	  \
	  + a02 * a[idx-n+1] + a12 * a[idx+1] + a22 * a[idx+n+1];


      } else  if (i == 0 && j ==0) {
	c[idx] =  a11 * a[idx]   + a21 * a[idx+n]	  \
	  + a12 * a[idx+1] + a22 * a[idx+n+1];

      } else if ( i==0 && j==n-1 ) {
	c[idx] = a10 * a[idx-1]   + a20 * a[idx+n-1]  \
	  + a11 * a[idx]     + a21 * a[idx+n];

      } else if ( i==n-1 && j==0 ) {
	c[idx] = a01 * a[idx-n]   + a11 * a[idx] 	  \
	  + a02 * a[idx-n+1] + a12 * a[idx+1];

      } else if ( i==n-1 && j==n-1 ) {
	c[idx] = a00 * a[idx-n-1] + a10 * a[idx-1] \
	  + a01 * a[idx-n]   + a11 * a[idx];

      } else if ( i==0 && 0<j && j<n-1) {
	c[idx] =  a10 * a[idx-1] + a20 * a[idx+n-1] \
	  + a11 * a[idx]   + a21 * a[idx+n]	\
	  + a12 * a[idx+1] + a22 * a[idx+n+1];

      } else if ( i==n-1 && 0<j && j<n-1) {
	c[idx] = a00 * a[idx-n-1] + a10 * a[idx-1] \
	  + a01 * a[idx-n]   + a11 * a[idx]   \
	  + a02 * a[idx-n+1] + a12 * a[idx+1];

      } else if (j==0 && 0<i && i<n-1) {
	c[idx] = a01 * a[idx-n]   + a11 * a[idx]   + a21 * a[idx+n]	  \
	  + a02 * a[idx-n+1] + a12 * a[idx+1] + a22 * a[idx+n+1];

      } else if (j==n-1 && 0<i && i<n-1) {
	c[idx] = a00 * a[idx-n-1] + a10 * a[idx-1] + a20 * a[idx+n-1] \
	  + a01 * a[idx-n]   + a11 * a[idx]   + a21 * a[idx+n];
      }
    }
  }
}




//------------------------------------------------------------------------------
// Main program
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int n = 10000;

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

  int loops = 10;

  for (i=0; i<loops; ++i) {
    conv_cpu(h_A, h_C, n, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  }


}
