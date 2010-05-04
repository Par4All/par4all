#include <stdio.h>
void foo_l(int a[4], int b[4], int c[4])
{
    int tmp[4];
    tmp[0]=b[0]*c[0];
    tmp[1]=b[1]*c[2];
    a[0]=tmp[0]+tmp[1];

    tmp[2]=b[0]*c[1];
    tmp[3]=b[1]*c[3];
    a[1]=tmp[2]+tmp[3];
}

int main() {
    int i,a[4],b[4],c[4];
    foo_l(a,b,c);
    for(i=0;i<4;i++)
        printf("%d",a[i]);
    return 0;
}

