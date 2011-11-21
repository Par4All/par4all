#include <stdio.h>
int main() {
    register int r1 asm("%eax") = 1;
    register int r2 asm("%ecx");
    asm("movl %eax, %ecx");
    printf("%d\n",r1);
    return 0;
}
