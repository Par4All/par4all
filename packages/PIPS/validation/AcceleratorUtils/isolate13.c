#include <stdio.h>
#define alpha 10
typedef struct {
    int a,b;
} ab;
void test(int n, ab src[n])
{
    unsigned int i,j;
    for(i=0;i<n;i++)
        src[i].a=src[i].b=3;
}

int main(int argc, char * argv[])
{
    int n = atoi(argv[1]);// yes this is unreliable
    ab a[n];
    int i;
    for(i=0;i<n;i++)
            a[i].a=1+ (a[i].b =1); 
    test(n,a);
    for(i=0;i<n;i++){
            printf("%d ",a[i].a);
        puts("\n");
    }
    return 0;
}
