#include <stdio.h>
#define MOD(a,b) ((a)%(b))
#define MAX0(a,b) (((a)>(b))?(a):(b))
typedef struct {
   float re;
   float im;
} Cplfloat;
typedef float v4sf[4];
typedef double v2df[2];

void average_power(int Nth, int Nrg, int Nv, Cplfloat ptrin[Nth][Nrg][Nv], Cplfloat Pow[Nth])
{
   int v;
   //PIPS generated variable
   float RED1[2], PP0;
   //PIPS generated variable
   int th0, v0;
   //PIPS generated variable
   v2df vec0;

         RED1[0] = 0.000000;
         RED1[1] = 0.000000;
         for(v = 4*(Nv/4); v <= Nv-1; v += 1) {
            //PIPS:SAC generated v2df vector(s)
            SIMD_LOAD_V2SF_TO_V2DF(vec0, &ptrin[0][0][v].re);
            SIMD_STORE_GENERIC_V2DF(vec0, &RED1[1], &RED1[0]);
         }
         PP0 = RED1[1]+RED1[0];
}
