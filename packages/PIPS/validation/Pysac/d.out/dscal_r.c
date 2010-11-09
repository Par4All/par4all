
/* Header automatically inserted by PYPS for defining MAX, MIN, MOD and others */
#ifndef MAX0
# define MAX0(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MAX
# define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
# define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MOD
# define MOD(a, b) ((a) % (b))
#endif

#ifndef DBLE
# define DBLE(a) ((double)(a))
#endif

#ifndef INT
# define INT(a) ((int)(a))
#endif

#ifdef WITH_TRIGO
#  include <math.h>
#  ifndef COS
#    define COS(a) (cos(a))
#  endif

#  ifndef SIN
#    define SIN(a) (sin(a))
#  endif
#endif
/* End header automatically inserted by PYPS for defining MAX, MIN, MOD and others */


#include <xmmintrin.h>

typedef float  a2sf[2] __attribute__ ((aligned (16)));
typedef float  a4sf[4] __attribute__ ((aligned (16)));
typedef double a2df[2] __attribute__ ((aligned (16)));
typedef int	a4si[4] __attribute__ ((aligned (16)));

typedef __m128  v4sf;
typedef __m128d v2df;
typedef __m128i v4si;
typedef __m128i v8hi;

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=_mm_loadu_ps(arr)
#define SIMD_LOADA_V4SF(vec,arr) vec=_mm_load_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm_mul_ps(vec2,vec3)
#define SIMD_DIVPS(vec1,vec2,vec3) vec1=_mm_div_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm_add_ps(vec2,vec3)
#define SIMD_SUBPS(vec1, vec2, vec3) vec1 = _mm_sub_ps(vec2, vec3)
/* umin as in unary minus */
#define SIMD_UMINPS(vec1, vec2)						do {								__m128 __pips_tmp;					__pips_tmp = _mm_setzero_ps();				vec1 = _mm_sub_ps(__pips_tmp, vec2);			} while(0)

#define SIMD_STORE_V4SF(vec,arr) _mm_storeu_ps(arr,vec)
#define SIMD_STOREA_V4SF(vec,arr) _mm_store_ps(arr,vec)
#define SIMD_STORE_GENERIC_V4SF(vec,v0,v1,v2,v3)					do {										float __pips_tmp[4] __attribute__ ((aligned (16)));			SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);					*(v0)=__pips_tmp[0];							*(v1)=__pips_tmp[1];							*(v2)=__pips_tmp[2];							*(v3)=__pips_tmp[3];							} while (0)

#define SIMD_ZERO_V4SF(vec) vec = _mm_setzero_ps()

#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)						do {										float __pips_v[4] __attribute ((aligned (16)));		__pips_v[0]=v0;		__pips_v[1]=v1;		__pips_v[2]=v2;		__pips_v[3]=v3;		SIMD_LOADA_V4SF(vec,&__pips_v[0]);					} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_STORE_MASKED_V4SF(vec,arr)							do {										float __pips_tmp[4] __attribute__ ((aligned (16)));							SIMD_STOREA_V4SF(vec,&__pips_tmp[0]);					(arr)[0] = __pips_tmp[0];						(arr)[1] = __pips_tmp[1];						(arr)[2] = __pips_tmp[2];						} while(0)

#define SIMD_LOAD_V4SI_TO_V4SF(v, f)				do {							float __pips_tmp[4];				__pips_tmp[0] = (f)[0];				__pips_tmp[1] = (f)[1];				__pips_tmp[2] = (f)[2];				__pips_tmp[3] = (f)[3];				SIMD_LOAD_V4SF(v, __pips_tmp);			} while(0)

/* double */
#define SIMD_LOAD_V2DF(vec,arr) vec=_mm_loadu_pd(arr)
#define SIMD_MULPD(vec1,vec2,vec3) vec1=_mm_mul_pd(vec2,vec3)
#define SIMD_ADDPD(vec1,vec2,vec3) vec1=_mm_add_pd(vec2,vec3)
#define SIMD_UMINPD(vec1, vec2)						do {								__m128d __pips_tmp;					__pips_tmp = _mm_setzero_pd();				vec1 = _mm_sub_pd(__pips_tmp, vec2);			} while(0)

#define SIMD_COSPD(vec1, vec2)								do {										double __pips_tmp[2] __attribute__ ((aligned (16)));			SIMD_STORE_V2DF(vec2, __pips_tmp);					__pips_tmp[0] = cos(__pips_tmp[0]);					__pips_tmp[1] = cos(__pips_tmp[1]);					SIMD_LOAD_V2DF(vec2, __pips_tmp);					} while(0)

#define SIMD_SINPD(vec1, vec2)								do {										double __pips_tmp[2] __attribute__ ((aligned (16)));			SIMD_STORE_V2DF(vec2, __pips_tmp);					__pips_tmp[0] = sin(__pips_tmp[0]);					__pips_tmp[1] = sin(__pips_tmp[1]);					} while(0)

#define SIMD_STORE_V2DF(vec,arr) _mm_storeu_pd(arr,vec)
#define SIMD_STORE_GENERIC_V2DF(vec, v0, v1)			do {							double __pips_tmp[2];					SIMD_STORE_V2DF(vec,&__pips_tmp[0]);			*(v0)=__pips_tmp[0];					*(v1)=__pips_tmp[1];					} while (0)
#define SIMD_LOAD_GENERIC_V2DF(vec,v0,v1)			do {							double v[2] = { v0,v1};				SIMD_LOAD_V2DF(vec,&v[0]);			} while(0)

/* conversions */
#define SIMD_STORE_V2DF_TO_V2SF(vec,f)					do {								double __pips_tmp[2];					SIMD_STORE_V2DF(vec, __pips_tmp);			(f)[0] = __pips_tmp[0];					(f)[1] = __pips_tmp[1];					} while(0)

#define SIMD_LOAD_V2SF_TO_V2DF(vec,f)				SIMD_LOAD_GENERIC_V2DF(vec,(f)[0],(f)[1])

/* char */
#define SIMD_LOAD_V8HI(vec,arr) 		vec = (__m128i*)(arr)

#define SIMD_STORE_V8HI(vec,arr)		*(__m128i *)(&(arr)[0]) = vec

#define SIMD_STORE_V8HI_TO_V8SI(vec,arr)	SIMD_STORE_V8HI(vec,arr)
#define SIMD_LOAD_V8SI_TO_V8HI(vec,arr)	SIMD_LOAD_V8HI(vec,arr)



/*
 * file for dscal_r.c
 */
/* PIPS include guard begin: #include <stdio.h> */
#include <stdio.h>
/* PIPS include guard end: #include <stdio.h> */
/* PIPS include guard begin: #include <stdlib.h> */
#include <stdlib.h>
/* PIPS include guard end: #include <stdlib.h> */


void dscal_r(int n, float da, float dx[n]);

int main(int argc, char *argv[]);
void dscal_r(int n, float da, float dx[n])
{
   //PIPS generated variable
   int i0, i1;
   //SAC generated temporary array
   a4sf pdata0 = {da, da, da, da};
   //PIPS generated variable
   v4sf vec00_0, vec10_0;
   SIMD_LOAD_V4SF(vec10_0, &pdata0[0]);
   for(i0 = 0; i0 <= 4*(n/4)-1; i0 += 4) {
      //PIPS:SAC generated v4sf vector(s)
      SIMD_LOAD_V4SF(vec00_0, &dx[i0]);
      SIMD_MULPS(vec00_0, vec10_0, vec00_0);
      SIMD_STORE_V4SF(vec00_0, &dx[i0]);
   }
   for(i1 = 4*(n/4); i1 <= n-1; i1 += 1)
      dx[i1] = pdata0[0]*dx[i1];
   ;
}
int main(int argc, char *argv[])
{
   int n = argc==1?10:atoi(argv[1]);
   float dx[n];
   int i;
   for(i = 0; i <= n-1; i += 1)
      dx[i] = i;
   dscal_r(n, 42., dx);
   for(i = 0; i <= n-1; i += 1)
      printf("%f", dx[i]);
   return 0;
}
