typedef struct {
    int a;
    int b[10];
    int *c;
} param_t;

int pain() {
    param_t p = { 1, { 0,1,2,3,4,5,6,7,8,9}, malloc(sizeof(int)*100) };
holy: if(1) {
             p.a=p.b[3]*p.c[8];
         }
         return p.a;
}
#include <stdio.h>
int main() {
    printf("%d\n",pain());
    return 0;
}
