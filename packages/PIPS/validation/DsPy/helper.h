/* helper.c */
#ifdef __CUDACC__
#include <p4a_complex.h>
#define fcomplex p4a_complexf
#else
#include <complex.h>
#define fcomplex float complex
#endif
void finit(int n, float *a);
void cshow(int n, fcomplex *a);
void fshow(int n, float *a);
