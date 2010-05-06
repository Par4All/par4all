#include <stdio.h>
int main()
{
    int i;
    int a,b,c;
    for(i=0;i<10;i++)
    {
        a=1;
        a=2;
        b=a;
        a=3;
        c=a;
        printf("%d-%d-%d",a,b,c);
    }
    return 0;
}
