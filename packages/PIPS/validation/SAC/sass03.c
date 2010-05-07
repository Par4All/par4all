#include <stdio.h>
int main()
{
    int a,b,c;
    a=1;
    a=2;
    b=a;
    b=b+a;
    a=3;
    c=a;
    printf("%d-%d-%d",a,b,c);
    return 0;
}
