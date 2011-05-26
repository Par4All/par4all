#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

void ddot_i(int n,int b[n], int c[n], int* r)
{
    int i;
    int a=0;
    for(i=0; i<n; ++i)
        a += b[i] * c[i] ;
    *r = a;
}

int main(int argc, char ** argv)
{
    int i,n;
    int a, (*b)[n], (*c)[n];
    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    b = malloc(n * sizeof(int));
    c = malloc(n * sizeof(int));
    init_data_file(argv[2]);
    init_data_int(b,n);
    init_data_int(c,n);
    close_data_file();
    ddot_i(n,*b, *c, &a);
    printf("%d\n",a);
    free(b);
    free(c);
    return 0;
}

