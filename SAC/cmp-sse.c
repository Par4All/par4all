#include <xmmintrin.h>

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
/*
 * file for cmp.c
 */
extern void cmp(float data[128], float val);                            /*0001*/

extern int main();                                                      /*0003*/
void cmp(float data[128], float val)
{
   int i, j[127+1];                                                     /*0004*/
   //PIPS generated variable
   int LU_NUB0, LU_IB0, LU_IND0;
   //PIPS:SAC generated int vector(s)
   int vec0;
   //PIPS:SAC generated float vector(s)
   __m128 vec1, vec2;
   LU_NUB0 = 128;
   LU_IB0 = 0;
   LU_IND0 = 0;
   SIMD_LOAD_GENERIC_V4SF(vec2, val, val, val, val);
   for(LU_IND0 = 0; LU_IND0 <= 127; LU_IND0 += 4) {
      SIMD_LOAD_V4SF(vec1, &data[LU_IND0]);
      SIMD_GTPS(vec0, vec1, vec2);
      SIMD_SAVE_V4SI(vec0, &j[LU_IND0]);
   }
   i = 128;
   ;
}
int main()
{
   float data[128];                                                     /*0004*/
   int i;                                                               /*0005*/
   for(i = 0; i <= 127; i += 1)
      data[i] = i;                                                      /*0007*/
   cmp(data, 10.f);                                                     /*0008*/
   for(i = 0; i <= 127; i += 1)
      printf("%d", i);                                                  /*0010*/
   return 0;                                                            /*0011*/
}
