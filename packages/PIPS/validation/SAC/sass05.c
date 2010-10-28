#include <stdio.h>
int main()
{
    int a=0,b,c;
    a=a+1;
    b=a;
    c=a;
    c=b;
    a=a+2;
    printf("%d",a);
    printf("%d",a);
    return 0;
}
