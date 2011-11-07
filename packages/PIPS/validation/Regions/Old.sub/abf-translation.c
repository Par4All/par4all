#include <stdio.h>


typedef struct {
  float   re;
  float   im;
} Cplfloat;

Cplfloat IN[2000][64][19];
Cplfloat OUT[2000][64][18];

void MTI ( int pul, int ant, Cplfloat Vin[ant][pul], Cplfloat Vout[ant][pul-1])
{
  int t,p;
  for (p=0; p<ant; p++) {
    for (t=1; t<pul; t++) {
      Vout[p][t-1].re= Vin[p][t].re - Vin[p][t-1].re;
      Vout[p][t-1].im= Vin[p][t].im - Vin[p][t-1].im;
    }
  }
}

void main_PE0(){
  
  int i =0;
  
  for (i =0; i<2000; i++)
    MTI(19,64, IN[i], OUT[i]);
  
}
