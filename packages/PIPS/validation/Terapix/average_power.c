int N;
#include<stdio.h>
#include<math.h>
// Define complex number
typedef struct {
 float re;
 float im;
} Cplfloat;

float CplAbs(Cplfloat const * c) {
    return sqrtf(c->re*c->re+c->im*c->im);
}


void average_power(int Nth, int Nrg, int Nv, Cplfloat ptrin[Nth][Nrg][Nv], float Pow[Nth]) {
 int th, v, rg;
 for(th=0;th<Nth;++th)
  for (rg=0; rg<Nrg; rg++)
   for (v=0; v<Nv; v++)
    Pow[th]+=CplAbs(&ptrin[th][rg][v]);
}

int main(int argc, char *argv[])
{
    int i,j,k;
    int th,rg,v;
    th=12,rg=13,v=v;
    {
        Cplfloat in[th][rg][v];
        float pow[th];
        for(i=0;i<th;i++) {
            pow[th]=0.;
            for(j=0;j<rg;j++)
                for(k=0;k<v;k++)
                {
                    in[i][j][k].re=i*j*k;
                    in[i][j][k].im=i*j+k;
                }
        }
        average_power(th,rg,v,in,pow);
        for(i=0;i<th;i++) 
            pow[th]/=rg*v;
        /* only print with bad precision for validation */
        for(i=0;i<th;i++)
            printf("-%d-%d-", ((int)pow[i])/10);
    }
    return 0;
}

