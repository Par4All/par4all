
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
 * file for ddot_ur.c
 */
/* PIPS include guard begin: #include <stdlib.h> */
#include <stdlib.h>
/* PIPS include guard end: #include <stdlib.h> */
/* PIPS include guard begin: #include <stdio.h> */
#include <stdio.h>
/* PIPS include guard end: #include <stdio.h> */

float ddot_ur(int n, float dx[n], float dy[n]);

int main(int argc, char **argv);
float ddot_ur(int n, float dx[n], float dy[n])
{
   float dtemp = 0;
   //PIPS generated variable
   float F_0, F_2, F_4, F_6, F_8, F_9;
   //PIPS generated variable
   float RED0[4], RED1[1];
   //PIPS generated variable
   int LU_IND00, LU_IND01, LU_IB00, LU_NUB00;
   //PIPS generated variable
   float F_80, F_81, F_82, F_83, F_60, F_61, F_62, F_63, F_40, F_41, F_42, F_43, F_20, F_21, F_22, F_23;
   //PIPS generated variable
   int i0, i1, m0;
   //PIPS generated variable
   float dtemp0, dtemp1, dtemp2, dtemp3;
   //SAC generated temporary array
   a4sf pdata1 = {0, 0, 0, 0}, pdata2 = {0, 0, 0, 0}, pdata3 = {0, 0, 0, 0}, pdata4 = {0, 0, 0, 0}, pdata5 = {0, 0, 0, 0}, pdata6 = {0, 0, 0, 0};
   //PIPS generated variable
   v4sf vec00_0, vec10_0, vec20_0, vec30_0, vec50_0, vec60_0, vec70_0, vec80_0, vec90_0, vec100_0, vec110_0, vec120_0, vec130_0, vec140_0, vec150_0, vec160_0, vec170_0, vec180_0, vec190_0, vec200_0, vec210_0, vec220_0;
   m0 = n%5;
   if (m0!=0) {
      RED0[0] = 0.000000;
      RED0[1] = 0.000000;
      RED0[2] = 0.000000;
      RED0[3] = 0.000000;
      SIMD_LOAD_V4SF(vec30_0, &RED0[0]);
      for(i0 = 0; i0 <= 4*(m0/4)-1; i0 += 4) {
         //PIPS:SAC generated v4sf vector(s)
         SIMD_LOAD_V4SF(vec20_0, &dy[i0]);
         SIMD_LOAD_V4SF(vec10_0, &dx[i0]);
         SIMD_MULPS(vec00_0, vec10_0, vec20_0);
         SIMD_ADDPS(vec30_0, vec30_0, vec00_0);
      }
      SIMD_STORE_V4SF(vec30_0, &RED0[0]);
      dtemp = RED0[3]+RED0[2]+RED0[1]+RED0[0]+dtemp;
      RED1[0] = 0.000000;
      for(i1 = 4*(m0/4); i1 <= m0-1; i1 += 1) {
         F_0 = dx[i1]*dy[i1];
         RED1[0] = RED1[0]+F_0;
      }
      dtemp = RED1[0]+dtemp;
      if (n<5) 
         return dtemp;
   }
   LU_NUB00 = (4+n-m0)/5;
   LU_IB00 = MOD(LU_NUB00, 4);
   for(LU_IND00 = 0; LU_IND00 <= LU_IB00-1; LU_IND00 += 1) {
      //PIPS:SAC generated v4sf vector(s)
      SIMD_LOAD_V4SF(vec70_0, &dy[m0+5*LU_IND00]);
      SIMD_LOAD_V4SF(vec60_0, &dx[m0+5*LU_IND00]);
      SIMD_MULPS(vec50_0, vec60_0, vec70_0);
      SIMD_STORE_V4SF(vec50_0, &pdata1[0]);
      pdata1[0] = dx[m0+5*LU_IND00]*dy[m0+5*LU_IND00];
      F_9 = dx[4+m0+5*LU_IND00]*dy[4+m0+5*LU_IND00];
      F_2 = dtemp+pdata1[0];
      F_4 = F_2+pdata1[1];
      F_6 = F_4+pdata1[2];
      F_8 = F_6+pdata1[3];
      dtemp = F_8+F_9;
   }
   for(LU_IND01 = LU_IB00; LU_IND01 <= LU_NUB00-1; LU_IND01 += 4) {
      //PIPS:SAC generated v4sf vector(s)
      SIMD_LOAD_V4SF(vec100_0, &dy[m0+5*LU_IND01]);
      SIMD_LOAD_V4SF(vec90_0, &dx[m0+5*LU_IND01]);
      SIMD_MULPS(vec80_0, vec90_0, vec100_0);
      SIMD_STORE_V4SF(vec80_0, &pdata2[0]);
      SIMD_LOAD_V4SF(vec130_0, &dy[4+m0+5*LU_IND01]);
      SIMD_LOAD_V4SF(vec120_0, &dx[4+m0+5*LU_IND01]);
      SIMD_MULPS(vec110_0, vec120_0, vec130_0);
      SIMD_STORE_V4SF(vec110_0, &pdata3[0]);
      SIMD_LOAD_V4SF(vec160_0, &dy[8+m0+5*LU_IND01]);
      SIMD_LOAD_V4SF(vec150_0, &dx[8+m0+5*LU_IND01]);
      SIMD_MULPS(vec140_0, vec150_0, vec160_0);
      SIMD_STORE_V4SF(vec140_0, &pdata4[0]);
      SIMD_LOAD_V4SF(vec190_0, &dy[12+m0+5*LU_IND01]);
      SIMD_LOAD_V4SF(vec180_0, &dx[12+m0+5*LU_IND01]);
      SIMD_MULPS(vec170_0, vec180_0, vec190_0);
      SIMD_STORE_V4SF(vec170_0, &pdata5[0]);
      SIMD_LOAD_V4SF(vec220_0, &dy[16+m0+5*LU_IND01]);
      SIMD_LOAD_V4SF(vec210_0, &dx[16+m0+5*LU_IND01]);
      SIMD_MULPS(vec200_0, vec210_0, vec220_0);
      SIMD_STORE_V4SF(vec200_0, &pdata6[0]);
      F_20 = dtemp3+pdata2[0];
      F_40 = F_20+pdata2[1];
      F_60 = F_40+pdata2[2];
      F_80 = F_60+pdata2[3];
      dtemp0 = F_80+pdata3[0];
      F_21 = dtemp0+pdata3[1];
      F_41 = F_21+pdata3[2];
      F_61 = F_41+pdata3[3];
      F_81 = F_61+pdata4[0];
      dtemp1 = F_81+pdata4[1];
      F_22 = dtemp1+pdata4[2];
      F_42 = F_22+pdata4[3];
      F_62 = F_42+pdata5[0];
      F_82 = F_62+pdata5[1];
      dtemp2 = F_82+pdata5[2];
      F_23 = dtemp2+pdata5[3];
      F_43 = F_23+pdata6[0];
      F_63 = F_43+pdata6[1];
      F_83 = F_63+pdata6[2];
      dtemp3 = F_83+pdata6[3];
   }

   return dtemp3;
}
int main(int argc, char **argv)
{
   int i, n = argc==1?200:atoi(argv[1]);
   float a, (*b)[n], (*c)[n];
   b = malloc(n*sizeof(float));
   c = malloc(n*sizeof(float));
   for(i = 0; i <= n-1; i += 1) {
      (*b)[i] = (float) i;
      (*c)[i] = (float) i;
   }
   a = ddot_ur(n, *b, *c);
   printf("%f", a);
   free(b);
   free(c);
   return 0;
}
