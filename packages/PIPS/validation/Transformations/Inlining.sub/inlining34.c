#include <stdio.h>

int foo(int n) {
    printf("%d.",n);
    return n;
}

int main() {
    int a[1] = { 3 };
    int *b=&a[0];
    foo(0);
    foo(sizeof(char));
    foo((b+0)[0]);
    foo(a[0]);
    return 0;
}
