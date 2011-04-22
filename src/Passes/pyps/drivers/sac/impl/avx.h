#include <immintrin.h>


typedef double		a4df[4] __attribute__ ((aligned (32)));
typedef float		a8sf[8] __attribute__ ((aligned (32)));
typedef float		a4sf[4] __attribute__ ((aligned (32)));
typedef long long	a4di[4] __attribute__ ((aligned (32)));
typedef int		a8si[8] __attribute__ ((aligned (32)));
typedef short		a16hi[16] __attribute__ ((aligned (32)));
typedef char		a32qi[32] __attribute__ ((aligned (32)));

typedef __m256d	v4df;
typedef __m256	v8sf;
typedef __m128  v4sf;
typedef __m256i	v4di;
typedef __m256i	v8si;
typedef __m256i	v16hi;
typedef __m256i	v32qi;

/* float */
#define SIMD_LOAD_V8SF(vec,arr) vec=_mm256_loadu_ps(arr)
#define SIMD_LOAD_BROADCAST_V8SF(vec,arr) vec=_mm256_set1_ps(arr)
#define SIMD_LOAD_BROADCAST_V4DF(vec,arr) vec=_mm256_set1_pd(arr)
#define SIMD_LOADA_V8SF(vec,arr) vec=_mm256_load_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm256_mul_ps(vec2,vec3)
#define SIMD_DIVPS(vec1,vec2,vec3) vec1=_mm256_div_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm256_add_ps(vec2,vec3)
#define SIMD_SUBPS(vec1, vec2, vec3) vec1 = _mm256_sub_ps(vec2, vec3)
#define SIMD_MULADDPS(vec1, vec2, vec3, vec4) \
		do { \
		__m256 __pips_tmp;\
		SIMD_MULPS(__pips_tmp,vec3,vec4); \
		SIMD_ADDPS(vec1,__pips_tmp,vec2); \
		} while(0)

#define SIMD_SHUFFLE_V8SF(dist,src,i0,i1,i2,i3) _mm256_shuffle_pd(src,src,_MM_SHUFFLE(i3,i2,i1,i0))
#define SIMD_SHUFFLE_V4SF(dist,src,i0,i1,i2,i3) _mm256_shuffle_ps(src,src,_MM_SHUFFLE(i3,i2,i1,i0))


/* umin as in unary minus */
#define SIMD_UMINPS(vec1, vec2)				\
		do {						\
		__m256 __pips_tmp;			\
		__pips_tmp = _mm256_setzero_ps();		\
		vec1 = _mm256_sub_ps(__pips_tmp, vec2);	\
		} while(0)

#define SIMD_STORE_V8SF(vec,arr) _mm256_storeu_ps(arr,vec)
#define SIMD_STOREA_V8SF(vec,arr) _mm256_store_ps(arr,vec)
#define SIMD_STORE_GENERIC_V8SF(vec,v0,v1,v2,v3,v4,v5,v6,v7)			\
		do {								\
		float __pips_tmp[4] __attribute__ ((aligned (32)));	\
		SIMD_STOREA_V8SF(vec,&__pips_tmp[0]);			\
		*(v0)=__pips_tmp[0];					\
		*(v1)=__pips_tmp[1];					\
		*(v2)=__pips_tmp[2];					\
		*(v3)=__pips_tmp[3];					\
		*(v4)=__pips_tmp[4];					\
		*(v5)=__pips_tmp[5];					\
		*(v6)=__pips_tmp[6];					\
		*(v7)=__pips_tmp[7];					\
		} while (0)

#define SIMD_ZERO_V8SF(vec) vec = _mm256_setzero_ps()
#define SIMD_LOAD_GENERIC_V8SF(vec,v0,v1,v2,v3,v4,v5,v6,v7)				\
		do {								\
		float __pips_v[8] __attribute ((aligned (32)));\
		vec=_mm256_set_ps(v0,v1,v2,v3,v4,v5,v6,v7);\
		} while(0)

#define SIMD_LOAD_V8SI_TO_V8SF(v, f)		\
		do {					\
		float __pips_tmp[8];		\
		__pips_tmp[0] = (f)[0];		\
		__pips_tmp[1] = (f)[1];		\
		__pips_tmp[2] = (f)[2];		\
		__pips_tmp[3] = (f)[3];		\
		__pips_tmp[4] = (f)[4];		\
		__pips_tmp[5] = (f)[5];		\
		__pips_tmp[6] = (f)[6];		\
		__pips_tmp[7] = (f)[7];		\
		SIMD_LOAD_V8SF(v, __pips_tmp);	\
		} while(0)

/* double */
#define SIMD_LOAD_V4DF(vec,arr) vec=_mm256_loadu_pd(arr)
#define SIMD_MULPD(vec1,vec2,vec3) vec1=_mm256_mul_pd(vec2,vec3)
#define SIMD_ADDPD(vec1,vec2,vec3) vec1=_mm256_add_pd(vec2,vec3)
#define SIMD_MULADDPD(vec1, vec2, vec3, vec4) \
		do { \
		__m256d __pips_tmp;\
		SIMD_MULPD(__pips_tmp,vec3,vec4); \
		SIMD_ADDPD(vec1,__pips_tmp,vec2); \
		} while(0)
#define SIMD_UMINPD(vec1, vec2)				\
		do {						\
		__m256d __pips_tmp;			\
		__pips_tmp = _mm256_setzero_pd();		\
		vec1 = _mm256_sub_pd(__pips_tmp, vec2);	\
		} while(0)

#define SIMD_COSPD(vec1, vec2)						\
		do {								\
		double __pips_tmp[4] __attribute__ ((aligned (16)));	\
		SIMD_STORE_V4DF(vec2, __pips_tmp);			\
		__pips_tmp[0] = cos(__pips_tmp[0]);			\
		__pips_tmp[1] = cos(__pips_tmp[1]);			\
		__pips_tmp[2] = cos(__pips_tmp[2]);			\
		__pips_tmp[3] = cos(__pips_tmp[3]);			\
		SIMD_LOAD_V4DF(vec1, __pips_tmp);			\
		} while(0)

#define SIMD_SINPD(vec1, vec2)						\
		do {								\
		double __pips_tmp[4] __attribute__ ((aligned (16)));	\
		SIMD_STORE_V4DF(vec2, __pips_tmp);			\
		__pips_tmp[0] = sin(__pips_tmp[0]);			\
		__pips_tmp[1] = sin(__pips_tmp[1]);			\
		__pips_tmp[2] = sin(__pips_tmp[2]);			\
		__pips_tmp[3] = sin(__pips_tmp[3]);			\
		SIMD_LOAD_V4DF(vec1, __pips_tmp);			\
		} while(0)

#define SIMD_STORE_V4DF(vec,arr) _mm256_storeu_pd(arr,vec)
#define SIMD_STORE_GENERIC_V4DF(vec, v0, v1, v2, v3)	\
		do {					\
		double __pips_tmp[4];			\
		SIMD_STORE_V4DF(vec,&__pips_tmp[0]);	\
		*(v0)=__pips_tmp[0];			\
		*(v1)=__pips_tmp[1];			\
		*(v2)=__pips_tmp[2];			\
		*(v3)=__pips_tmp[3];			\
		} while (0)

#define SIMD_LOAD_GENERIC_V4DF(vec,v0,v1,v2,v3)	\
		do {					\
		vec=_mm256_set_pd(v0,v1,v2,v3);\
		} while(0)

/* conversions */
#define SIMD_STORE_V4DF_TO_V4SF(vec,f)			\
		do {						\
		double __pips_tmp[4];			\
		SIMD_STORE_V4DF(vec, __pips_tmp);	\
		(f)[0] = __pips_tmp[0];			\
		(f)[1] = __pips_tmp[1];			\
		(f)[2] = __pips_tmp[2];			\
		(f)[3] = __pips_tmp[3];			\
		} while(0)

#define SIMD_LOAD_V4SF_TO_V4DF(vec,f)      \
    do {\
        __m128 vecsf = _mm_load_ps(f);\
        vec=_mm256_cvtps_pd(vecsf) ; \
    } while(0)

/* long long */
#define SIMD_LOADA_V4DI(vec,arr) \
		vec=_mm256_load_si256(arr)

#define SIMD_STOREA_V4DI(vec,arr)\
		vec=_mm256_store_si256(arr)

#define SIMD_LOAD_V4DI(vec,arr) \
		vec=_mm256_loadu_si256(arr)

#define SIMD_STORE_V4DI(vec,arr) \
		vec=_mm256_storeu_si256(arr)


/* int */
#define SIMD_LOADA_V8SI(vec,arr) \
		vec=_mm256_load_si256(arr)

#define SIMD_STOREA_V8SI(vec,arr)\
		vec=_mm256_store_si256(arr)

#define SIMD_LOAD_V8SI(vec,arr) \
		vec=_mm256_loadu_si256(arr)

#define SIMD_STORE_V8SI(vec,arr) \
		vec=_mm256_storeu_si256(arr)

/* short */
#define SIMD_LOADA_V16HI(vec,arr) \
		vec=_mm256_load_si256(arr)

#define SIMD_STOREA_V16HI(vec,arr)\
		vec=_mm256_store_si256(arr)

#define SIMD_LOAD_V16HI(vec,arr) \
		vec=_mm256_loadu_si256(arr)

#define SIMD_STORE_V16HI(vec,arr) \
		vec=_mm256_storeu_si256(arr)

/* char */
#define SIMD_LOADA_V32QI(vec,arr) \
		vec=_mm256_load_si256(arr)

#define SIMD_STOREA_V32QI(vec,arr)\
		vec=_mm256_store_si256(arr)

#define SIMD_LOAD_V32QI(vec,arr) \
		vec=_mm256_loadu_si256(arr)

#define SIMD_STORE_V32QI(vec,arr) \
		vec=_mm256_storeu_si256(arr)

