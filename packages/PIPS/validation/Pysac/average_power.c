#include<stdio.h>
#include <stdlib.h>
#include <err.h>

// Define complex number
typedef struct {
    float re;
    float im;
} Cplfloat;

void average_power(int Nth, int Nrg, int Nv, Cplfloat ptrin[Nth][Nrg][Nv], Cplfloat Pow[Nth]) {

    float PP;
    int th, v, rg;

    for (th=0; th<Nth; th++) {
        PP=0.;
        for (rg=0; rg<Nrg; rg++) {
            for (v=0; v<Nv; v++) {
                PP += ptrin[th][rg][v].re *ptrin[th][rg][v].re
                    +ptrin[th][rg][v].im *ptrin[th][rg][v].im;
            }
        }
        Pow[th].re= (float)(PP/((float)(Nv*Nrg)));
        Pow[th].im= 0.;
    }
}

int main(int argc, char *argv[])
{
    int i,j,k;
    int th,rg,v;
    if(argc < 42)
        th=256,rg=256,v=256;
    {
        Cplfloat (*in)[th][rg][v], (*pow)[th];
        in = malloc(th * rg * v *sizeof(Cplfloat));
        if (!in)
            err(1, "in = malloc(%zu)", th *rg*v* sizeof(Cplfloat));
        for (i=0; i<th; i++) {
            for (j=0; j<rg; j++) {
                for (k=0;k<v;k++) {
                    (*in)[i][j][k].re = i*j*k;
                    (*in)[i][j][k].re = i*j+k;
                }
            }
        }
        pow = malloc(th * sizeof(Cplfloat));
        if (!pow)
            err(1, "malloc(%zu)", th * sizeof(Cplfloat));
        average_power(th,rg,v,*in,*pow);
        /* only print with bad precision for validation */
        for(i=0;i<th;i++)
            printf("-%d-%d-\n", ((int)(*pow)[i].re)/10, ((int)(*pow)[i].im)/10);
    }
    return 0;
}

