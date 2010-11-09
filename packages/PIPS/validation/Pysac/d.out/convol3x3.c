
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
 * file for convol3x3.c
 */
//     goal: show effect of cloning, partial evaluation and loop unrolling
//     and reduction parallelization for a Power architecture
//     kernel_size must be odd
/* PIPS include guard begin: #include <stdio.h> */
#include <stdio.h>
/* PIPS include guard end: #include <stdio.h> */



void convol3x3(int isi, int isj, float new_image[isi][isj], float image[isi][isj], float kernel[3][3]);

int main(int argc, char **argv);

void convol3x3(int isi, int isj, float new_image[isi][isj], float image[isi][isj], float kernel[3][3]);
int main(int argc, char **argv)
{
   int image_size = atoi(argv[1]);
   float (*image)[image_size][image_size];
   float (*new_image)[image_size][image_size];
   float kernel[3][3];
   int i, j, n;
   image = malloc(sizeof(float)*image_size*image_size);
   new_image = malloc(sizeof(float)*image_size*image_size);
   
   
   for(i = 0; i <= 2; i += 1)
      for(j = 0; j <= 2; j += 1)
         kernel[i][j] = 1;
   
   //     read *, image
   for(i = 0; i <= image_size-1; i += 1)
      for(j = 0; j <= image_size-1; j += 1)
         ((*image)[i])[j] = 1.;
   
   
   for(n = 0; n <= 9; n += 1)

      convol3x3(image_size, image_size, *new_image, *image, kernel);

   for(i = 0; i <= image_size-1; i += 1)
      printf("%f ", ((*new_image)[i])[i]);
   free(image);
   free(new_image);
   //     print *, new_image
   //      print *, new_image (image_size/2, image_size/2)
   
   return 0;
}
void convol3x3(int isi, int isj, float new_image[isi][isj], float image[isi][isj], float kernel[3][3])
{
   //     The convolution kernel is not applied on the outer part
   //     of the image
   
   int j;
   //PIPS generated variable
   float F_8;
   //PIPS generated variable
   int i0, i1, j0;
   //SAC generated temporary array
   a4si pdata9 = {9, 9, 9, 9};
   //SAC generated temporary array
   a4sf pdata10 = {0, 0, 0, 0}, pdata11 = {0, 0, 0, 0};
   //PIPS generated variable
   v4sf vec00_0, vec10_0, vec20_0, vec30_0, vec40_0, vec50_0, vec60_0, vec70_0, vec80_0, vec90_0, vec100_0, vec110_0, vec120_0, vec130_0, vec140_0, vec150_0, vec160_0, vec170_0, vec180_0, vec190_0, vec200_0, vec210_0, vec220_0, vec230_0, vec240_0, vec250_0, vec260_0, vec270_0, vec290_0, vec310_0, vec330_0, vec350_0, vec370_0, vec390_0, vec410_0, vec430_0, vec450_0, vec460_0, vec470_0, vec480_0, vec490_0, vec500_0, vec510_0, vec520_0;

   for(i0 = 0; i0 <= isi-1; i0 += 1)
      for(j = 0; j <= isj-1; j += 1)
         new_image[i0][j] = image[i0][j];

   SIMD_LOAD_GENERIC_V4SF(vec20_0, kernel[0][0], kernel[0][0], kernel[0][0], kernel[0][0]);
   SIMD_LOAD_GENERIC_V4SF(vec50_0, kernel[0][1], kernel[0][1], kernel[0][1], kernel[0][1]);
   SIMD_LOAD_GENERIC_V4SF(vec80_0, kernel[0][2], kernel[0][2], kernel[0][2], kernel[0][2]);
   SIMD_LOAD_GENERIC_V4SF(vec110_0, kernel[1][0], kernel[1][0], kernel[1][0], kernel[1][0]);
   SIMD_LOAD_GENERIC_V4SF(vec140_0, kernel[1][1], kernel[1][1], kernel[1][1], kernel[1][1]);
   SIMD_LOAD_GENERIC_V4SF(vec170_0, kernel[1][2], kernel[1][2], kernel[1][2], kernel[1][2]);
   SIMD_LOAD_GENERIC_V4SF(vec200_0, kernel[2][0], kernel[2][0], kernel[2][0], kernel[2][0]);
   SIMD_LOAD_GENERIC_V4SF(vec230_0, kernel[2][1], kernel[2][1], kernel[2][1], kernel[2][1]);
   SIMD_LOAD_GENERIC_V4SF(vec260_0, kernel[2][2], kernel[2][2], kernel[2][2], kernel[2][2]);
   SIMD_LOAD_V4SI_TO_V4SF(vec460_0, &pdata9[0]);
   SIMD_LOAD_V4SF(vec490_0, &kernel[0][0]);
   SIMD_LOAD_V4SF(vec520_0, &kernel[1][1]);
   for(i1 = 1; i1 <= isi-2; i1 += 1) {
      for(j0 = 1; j0 <= 4*((isj-2)/4); j0 += 4) {
         //PIPS:SAC generated v4sf vector(s)
         new_image[i1][j0] = 0.;
         SIMD_LOAD_V4SF(vec10_0, &image[i1-1][j0-1]);
         SIMD_MULPS(vec00_0, vec10_0, vec20_0);
         SIMD_LOAD_V4SF(vec40_0, &image[i1-1][j0]);
         SIMD_MULPS(vec30_0, vec40_0, vec50_0);
         SIMD_LOAD_V4SF(vec70_0, &image[i1-1][1+j0]);
         SIMD_MULPS(vec60_0, vec70_0, vec80_0);
         SIMD_LOAD_V4SF(vec100_0, &image[i1][j0-1]);
         SIMD_MULPS(vec90_0, vec100_0, vec110_0);
         SIMD_LOAD_V4SF(vec130_0, &image[i1][j0]);
         SIMD_MULPS(vec120_0, vec130_0, vec140_0);
         SIMD_LOAD_V4SF(vec160_0, &image[i1][1+j0]);
         SIMD_MULPS(vec150_0, vec160_0, vec170_0);
         SIMD_LOAD_V4SF(vec190_0, &image[1+i1][j0-1]);
         SIMD_MULPS(vec180_0, vec190_0, vec200_0);
         SIMD_LOAD_V4SF(vec220_0, &image[1+i1][j0]);
         SIMD_MULPS(vec210_0, vec220_0, vec230_0);
         SIMD_LOAD_V4SF(vec250_0, &image[1+i1][1+j0]);
         SIMD_MULPS(vec240_0, vec250_0, vec260_0);
         new_image[i1][1+j0] = 0.;
         new_image[i1][2+j0] = 0.;
         new_image[i1][3+j0] = 0.;
         SIMD_LOAD_V4SF(vec270_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec270_0, vec270_0, vec00_0);
         SIMD_STORE_V4SF(vec270_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec290_0, vec270_0, vec30_0);
         SIMD_STORE_V4SF(vec290_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec310_0, vec290_0, vec60_0);
         SIMD_STORE_V4SF(vec310_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec330_0, vec310_0, vec90_0);
         SIMD_STORE_V4SF(vec330_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec350_0, vec330_0, vec120_0);
         SIMD_STORE_V4SF(vec350_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec370_0, vec350_0, vec150_0);
         SIMD_STORE_V4SF(vec370_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec390_0, vec370_0, vec180_0);
         SIMD_STORE_V4SF(vec390_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec410_0, vec390_0, vec210_0);
         SIMD_STORE_V4SF(vec410_0, &new_image[i1][j0]);
         SIMD_ADDPS(vec430_0, vec410_0, vec240_0);
         SIMD_STORE_V4SF(vec430_0, &new_image[i1][j0]);
         SIMD_DIVPS(vec450_0, vec430_0, vec460_0);
         SIMD_STORE_V4SF(vec450_0, &new_image[i1][j0]);
      }
      for(j = 1+4*((isj-2)/4); j <= isj-2; j += 1) {
         //PIPS:SAC generated v4sf vector(s)
         new_image[i1][j] = 0.;
         SIMD_LOAD_GENERIC_V4SF(vec480_0, image[i1-1][j-1], image[i1-1][j], image[i1-1][1+j], image[i1][j-1]);
         SIMD_MULPS(vec470_0, vec480_0, vec490_0);
         SIMD_STORE_V4SF(vec470_0, &pdata10[0]);
         SIMD_LOAD_GENERIC_V4SF(vec510_0, image[i1][j], image[i1][1+j], image[1+i1][j-1], image[1+i1][j]);
         SIMD_MULPS(vec500_0, vec510_0, vec520_0);
         SIMD_STORE_V4SF(vec500_0, &pdata11[0]);
         pdata10[0] = image[i1-1][j-1]*kernel[0][0];
         F_8 = image[1+i1][1+j]*kernel[2][2];
         new_image[i1][j] = new_image[i1][j]+pdata10[0];
         new_image[i1][j] = new_image[i1][j]+pdata10[1];
         new_image[i1][j] = new_image[i1][j]+pdata10[2];
         new_image[i1][j] = new_image[i1][j]+pdata10[3];
         new_image[i1][j] = new_image[i1][j]+pdata11[0];
         new_image[i1][j] = new_image[i1][j]+pdata11[1];
         new_image[i1][j] = new_image[i1][j]+pdata11[2];
         new_image[i1][j] = new_image[i1][j]+pdata11[3];
         new_image[i1][j] = new_image[i1][j]+F_8;
         new_image[i1][j] = new_image[i1][j]/9;
      }
   }
   ;
}
