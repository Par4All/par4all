#include <stdio.h>
int main() {
    int i;
    int a,b;
    a=b;
    for(i=0;i<2;i++) {
        a=3;
        b=a+4;
        b=a+4;
        printf("%d",b);
        a=3;
        b=a+4;
        printf("%d",b);
    }
    return 0;
}
