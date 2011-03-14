#ifndef P4A_ACCEL_COMPLEX_H
#define P4A_ACCEL_COMPLEX_H

#ifdef P4A_ACCEL_OPENMP
#include <complex.h>
#endif

#ifdef P4A_ACCEL_CUDA
#include <math.h>
#include "p4a_complex.h"

// For compatibility with the C99 expressions
#define complex p4a_complexf
#define creal(a) crealf(a)
#define cimag(a) cimagf(a)
#define cexp(a) cexpf(a)
#endif

#ifdef P4A_ACCEL_OPENCL
//#include <math.h>
#include "p4a_complex.h"
#define complex p4a_complexf
#define creal(a) crealf(a)
#define cimag(a) cimagf(a)
#define cexp(a) cexpf(a)

/*
typedef float2 complex;
#define I (float2)(0.f,1.f)
//inline float2 cexp(float2 a) {return (float2)(exp(a.x)*cos(a.y),exp(a.x)*sin(a.y));} 
inline float2 cexp(float2 a) {float2 b = (float2)(cos(a.y),sin(a.y)); return (float2)(exp(a.x)*b);}
*/ 
#endif

#endif //P4A_ACCEL_COMPLEX_H

