#include <stdio.h>

void foo_l(int a[4],int b[4],int c[4])
{
    int i,j;
    for(j=0;j<10;j++)
        for(i=0;i<10;i++)
        {
            a[0]=b[0]+c[0];
            a[1]=b[1]+c[1];
            a[2]=b[2]+c[2];
            a[3]=b[3]+c[3];
        }
}
