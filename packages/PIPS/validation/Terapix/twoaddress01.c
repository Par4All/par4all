#include<stdio.h>
int main()
{
    int a=2,b=1;
    int c=&a;
    c=a>>b;
    printf("%d",c);
    return 0;
}
