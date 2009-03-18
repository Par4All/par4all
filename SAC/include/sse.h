#include <xmmintrin.h>

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
