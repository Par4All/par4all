#include <stdio.h>
void foo_l(double a[4], double b[4], double c[4])
{
    double tmp[4];
    tmp[0]=b[0]*1.;
    tmp[1]=b[1]*2.;
    a[0]=tmp[0]+tmp[1];

    tmp[2]=b[0]*3;
    tmp[3]=b[1]*4;
    a[1]=tmp[2]+tmp[3];
}

int main() {
    double a[4],b[4],c[4];
    for(int i=0;i<4;i++)
        b[i]=i;
    foo_l(a,b,c);
    for(int i=0;i<4;i++)
        printf("%f",a[i]);
    return 0;
}


