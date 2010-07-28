
#define simd_phi(l,x1,x2) (l)?(x1):(x2)

#define simd_mul(to,from1,from2) (to).v = (from1).v*(from2).v
#define simd_add(to,from1,from2) (to).v = (from1).v+(from2).v
#define simd_sub(to,from1,from2) (to).v = (from1).v-(from2).v
#define simd_gt(to,from1,from2) (to).v = (from1).v < (from2).v
#define simd_phi(to,from1,from2,from3) (to).v = (from1).v ? (from2.v) : (from3).v

#define simd_load(to,from) memcpy(&(to.i),&(from),sizeof(to))
#define simd_save(from,to) memcpy(&(to),&(from.i),sizeof(to))

#define simd_load_generic(to, v1,v2,v3,v4) \
        do { \
                    (to).i[0]=(v1);\
                    (to).i[1]=(v2);\
                    (to).i[2]=(v3);\
                    (to).i[3]=(v4);\
                } while(0)
#define simd_save_generic(to, v1,v2,v3,v4) \
        do { \
                    (v1)=(to).i[0];\
                    (v2)=(to).i[1];\
                    (v3)=(to).i[2];\
                    (v4)=(to).i[3];\
                } while(0)

typedef union{
        short v __attribute__ ((vector_size (sizeof(short)*4)));
        short i[4];
} v4hi;

typedef union{
        float v __attribute__ ((vector_size (sizeof(float)*4)));
        float i[4];
} v4sf;
typedef union{
        int v __attribute__ ((vector_size (sizeof(int)*4)));
        int i[4];
} v4si;

#define SIMD(l) SIMD_l

void SIMD(LOAD_V4SF)( v4sf vec, float* base)
{
    simd_load(vec,base);
}

void SIMD(LOAD_V4HI)( v4hi vec, float* base)
{
    simd_load(vec,base);
}

void SIMD(LOAD_V4SI)( v4si vec, int* base)
{
    simd_load(vec,base);
}

void SIMD(LOAD_GENERIC_V4SF)(v4sf vec, float x1, float x2, float x3, float x4)
{
    simd_load_generic(vec,x1,x2,x3,x4);
}

void SIMD(LOAD_GENERIC_V4HI)(v4hi vec, float x1, float x2, float x3, float x4)
{
    simd_load_generic(vec,x1,x2,x3,x4);
}

void SIMD(LOAD_GENERIC_V4SI)(v4si vec, int x1, int x2, int x3, int x4)
{
    simd_load_generic(vec,x1,x2,x3,x4);
}

void SIMD(STORE_V4SI)(v4si vec, int *dest)
{
    simd_save(vec,dest);
}

#define simd_load_constant_v4hi simd_load_generic
#define simd_load_constant_v4sf simd_load_generic

#define simd_save_generic_v4hi simd_save_generic
#define simd_save_generic_v4sf simd_save_generic

#define simd_mulw  simd_mul
#define simd_mulps simd_mul

#define simd_addw  simd_add
#define simd_addps simd_add

#define simd_subw  simd_sub
#define simd_subps simd_sub

