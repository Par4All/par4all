#include <string.h>
#define simd_mul(to,from1,from2) (to).v = (from1).v*(from2).v
#define simd_add(to,from1,from2) (to).v = (from1).v+(from2).v
#define simd_sub(to,from1,from2) (to).v = (from1).v-(from2).v

#define simd_load(to,from) memcpy(&(to.i),&(from),sizeof(to))
#define simd_store(from,to) memcpy(&(to),&(from.i),sizeof(to))

#define simd_load_generic(to, v1,v2,v3,v4) \
    do { \
        (to).i[0]=(v1);\
        (to).i[1]=(v2);\
        (to).i[2]=(v3);\
        (to).i[3]=(v4);\
    } while(0);
#define simd_store_generic(to, v1,v2,v3,v4) \
    do { \
        (v1)=(to).i[0];\
        (v2)=(to).i[1];\
        (v3)=(to).i[2];\
        (v4)=(to).i[3];\
    } while(0);

#define simd_load_v4hi simd_load
#define simd_load_v4sf simd_load

#define simd_store_v4hi simd_store
#define simd_store_v4sf simd_store

#define simd_load_generic_v4hi simd_load_generic
#define simd_load_generic_v4sf simd_load_generic

#define simd_load_constant_v4hi simd_load_generic
#define simd_load_constant_v4sf simd_load_generic

#define simd_store_generic_v4hi simd_store_generic
#define simd_store_generic_v4sf simd_store_generic

#define simd_mulw  simd_mul
#define simd_mulps simd_mul

#define simd_addw  simd_add
#define simd_addps simd_add

#define simd_subw  simd_sub
#define simd_subps simd_sub


typedef union{
        short v __attribute__ ((vector_size (sizeof(short)*4)));
        short i[4];
} v4hi;

typedef union{
        float v __attribute__ ((vector_size (sizeof(float)*4)));
        float i[4];
} v4sf;

