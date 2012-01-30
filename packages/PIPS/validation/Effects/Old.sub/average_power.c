#include <stdio.h>
// Define complex number
typedef struct {
    float re;
    float im;
} Cplfloat;

void average_power(int Nth, int Nrg, int Nv, Cplfloat ptrin[Nth][Nrg][Nv],
        Cplfloat Pow[Nth]) {

    double PP;
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
    th=14,rg=13,v=12;
    {
        Cplfloat in[th][rg][v],pow[th];
        for(i=0;i<th;i++)
            for(j=0;j<rg;j++)
                for(k=0;k<v;k++)
                {
                    in[i][j][k].re=i*j*k;
                    in[i][j][k].im=i*j+k;
                }
        average_power(th,rg,v,in,pow);
        for(i=0;i<th;i++)
            printf("-%f-", pow[i]);
    }
    return 0;
}

