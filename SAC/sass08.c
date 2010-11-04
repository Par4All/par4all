#include <stdio.h>
main() {
    int i,a=0,b=1;
    for(i=0;i<10;i++) {
        b=a;
        b+=i;
        a=b;
    }
    printf("%d\n",a);
}
