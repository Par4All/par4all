#include <stdio.h>

/* KERNEL_ID is required by kernelize
 * a definition is not enough
 * we must declare it to get inter procedural effects
 */
int KERNEL_ID() {
    return random();
}

int foo(int seed[100]) {
    int a[100];
    int b[100];
    int c[100];
    int i;
    /* init */
    for(i=0;i<100;i++)
    {
        a[i]=seed[i];
        b[i]=seed[99-i];
    }
    /* compute */
kernel:    for(i=0;i<100;i++)
    {
        c[i]=a[i]+b[i];
    }
    /* print result */
    for(i=0;i<100;i++)
    {
        printf("%d ",c[i]);
    }
    printf("\n");
}
