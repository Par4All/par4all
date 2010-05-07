#include <stdio.h>
int main()
{
    int a,b,c,i;
    a=1;
    b=a;
    for(i=0;i<10;i++)
    {
        a=2;
        b=a+b;
        a=3;
    }
    c=a;
    printf("%d-%d-%d",a,b,c);
    return 0;
}
