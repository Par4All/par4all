#include <stdio.h>
#include <time.h>
// without the second loop or the test, it does not fail
int main(int arc, char **argv)
{
    int i,j,x;
loop:
    for (i = 0; i < 10; i++) {
        printf("%d\n", i);
    }

    if (i > 11)
        goto loop;

    x = 0;
    for (i = 0; i < 100; j++) {
        x += i;
    }
    return 0;
}


