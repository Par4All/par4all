#include <xmmintrin.h>

/* extras */
#define MOD(a,b) ((a)%(b))
#define MAX0(a,b) ((a)>(b)?(a):(b))

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=_mm_loadu_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm_mul_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm_add_ps(vec2,vec3)
#define SIMD_STORE_V4SF(vec,arr) _mm_storeu_ps(arr,vec)
#define SIMD_LOAD_BROADCAST_V4SF(vec,val) vec=_mm_set1_ps(val)
#define SIMD_SHUFFLE_V4SF(dist,src,i0,i1,i2,i3) dist=_mm_shuffle_ps(src,src,_MM_SHUFFLE(i3,i2,i1,i0))
#define SIMD_STORE_GENERIC_V4SF(vec,v0,v1,v2,v3) \
do { \
    float __tmp[4]; \
    SIMD_STORE_V4SF(vec,&__tmp[0]);\
    *v0=__tmp[0];\
    *v1=__tmp[1];\
    *v2=__tmp[2];\
    *v3=__tmp[3];\
} while (0)
#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)\
do { \
    float __tmp[4] = { v0,v1,v2,v3 };\
    SIMD_LOAD_V4SF(vec,&__tmp[0]); \
} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_STORE_MASKED_V4SF(vec,arr) do { float __tmp[4] ; SIMD_STORE_V4SF(vec,&__tmp[0]); (arr)[0]=__tmp[0];(arr)[1]=__tmp[1];(arr)[2]=__tmp[2]; } while(0)


/* double */
#define SIMD_LOAD_V2DF(vec,arr) vec=_mm_loadu_pd(arr)
#define SIMD_MULPD(vec1,vec2,vec3) vec1=_mm_mul_pd(vec2,vec3)
#define SIMD_ADDPD(vec1,vec2,vec3) vec1=_mm_add_pd(vec2,vec3)
#define SIMD_STORE_V2DF(vec,arr) _mm_storeu_pd(arr,vec)
#define SIMD_STORE_GENERIC_V2DF(vec,v0,v1) \
do { \
    double __tmp[2]; \
    SIMD_STORE_V2DF(vec,&__tmp[0]);\
    *(v0)=__tmp[0];\
    *(v1)=__tmp[1];\
} while (0)
#define SIMD_LOAD_GENERIC_V2DF(vec,v0,v1)\
do { \
    double __tmp[2] = { v0,v1};\
    SIMD_LOAD_V2DF(vec,&__tmp[0]); \
} while(0)

/* conversions */
#define SIMD_STORE_V2DF_TO_V2SF(f,vec) \
    SIMD_STORE_GENERIC_V2SF(vec,(f),(f)+1)
#define SIMD_LOAD_V2SF_TO_V2DF(vec,f) \
    SIMD_LOAD_GENERIC_V2DF(vec,(f)[0],(f)[1])


