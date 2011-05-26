#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

void ddot_uri(int n,int dx[n], int dy[n], int* r)
{
    int i,m;
    int dtemp=0;
    m = n % 5;
    if (m != 0)
    {
        for (i = 0; i < m; i++)
            dtemp = dtemp + dx[i]*dy[i];
        if (n < 5)
	{
		*r=dtemp;
		return;
	}
    }
    for (i = m; i < n; i = i + 5)
    {
        dtemp = dtemp + (dx[i]*dy[i] + dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2] + dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4]);
    }

    *r = dtemp;
}

int main(int argc, char ** argv)
{
    int i,n;
    int a, *b, *c;
    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    b = (int*) malloc(n * sizeof(int));
    c = (int*) malloc(n * sizeof(int));
    init_data_file(argv[2]);
    init_data_int(b,n);
    init_data_int(c,n);
    close_data_file();
    ddot_uri(n,b, c, &a);
    printf("%f\n",a);
    free(b);
    free(c);
}

