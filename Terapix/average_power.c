#include<stdio.h>
// Define complex number
typedef struct {
    float re;
    float im;
} Cplfloat;

static void internal(float *in, Cplfloat* from0, Cplfloat *from1)
{
    *in += from0->re * from1->re
        +from0->im * from1->im;
}

void average_power(int Nth, int Nrg, int Nv, Cplfloat ptrin[Nth][Nrg][Nv],
        Cplfloat Pow[Nth]) {

    int th, v, rg;

    for (th=0; th<Nth; th++) {
        for (rg=0; rg<Nrg; rg++) {
            for (v=0; v<Nv; v++) {
                internal(&(Pow[th].re),&(ptrin[th][rg][v]),&(ptrin[th][rg][v]));
            }
        }
        Pow[th].re/= (float)(Nv*Nrg);
        Pow[th].im= 0.;
    }
}

int main(int argc, char *argv[])
{
    int i,j,k;
    int th,rg,v;
    th=16,rg=13,v=12;
    {
        Cplfloat in[th][rg][v],pow[th];
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
        /* only print with bad precision for validation */
        for(i=0;i<th;i++)
            printf("-%d-%d-", ((int)pow[i].re)/10, ((int)pow[i].im))/10;
    }
    return 0;
}

