#include <stdio.h>
int main()
{
    int a[2]={0,1},(*b)[2],*c;
    b=&a;
    a[0]=a[1]+((*b)[1])+1 ;
    printf("%d",a[0]);
    return 0;
}
