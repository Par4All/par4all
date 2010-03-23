#include <xmmintrin.h>

/* extras */
#define MOD(a,b) ((a)%(b))
#define MAX0(a,b) ((a)>(b)?(a):(b))

/* float */
#define SIMD_LOAD_V4SF(vec,arr) vec=_mm_loadu_ps(arr)
#define SIMD_MULPS(vec1,vec2,vec3) vec1=_mm_mul_ps(vec2,vec3)
#define SIMD_ADDPS(vec1,vec2,vec3) vec1=_mm_add_ps(vec2,vec3)
#define SIMD_SAVE_V4SF(vec,arr) _mm_storeu_ps(arr,vec)
#define SIMD_SAVE_GENERIC_V4SF(vec,v0,v1,v2,v3) \
do { \
    float tmp[4]; \
    SIMD_SAVE_V4SF(vec,&tmp[0]);\
    *v0=tmp[0];\
    *v1=tmp[1];\
    *v2=tmp[2];\
    *v3=tmp[3];\
} while (0)
#define SIMD_LOAD_GENERIC_V4SF(vec,v0,v1,v2,v3)\
do { \
    float v[4] = { v0,v1,v2,v3 };\
    SIMD_LOAD_V4SF(vec,&v[0]); \
} while(0)

/* handle padded value, this is a very bad implementation ... */
#define SIMD_MASKED_SAVE_V4SF(vec,arr) do { float tmp[4] ; SIMD_SAVE_V4SF(vec,&tmp[0]); (arr)[0]=tmp[0];(arr)[1]=tmp[1];(arr)[2]=tmp[2]; } while(0)
