#include <stdio.h>
static void assign(int b[100], int i) {
    b[i]=i;
}

float ing(int i) {
    int b[100];
here:if(1) {
        printf("%d\n",i);
        for(i=0;i<100;i++) {
            assign(b,i);
        }
    }
}
