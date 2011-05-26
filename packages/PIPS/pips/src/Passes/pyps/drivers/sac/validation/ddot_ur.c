#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

void ddot_ur(int n,float dx[n], float dy[n], float* r)
{
    int i,m;
    float dtemp=0;
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
        dtemp = dtemp + (dx[i]*dy[i] +
            dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2] +
            dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4]);
    }

    *r = dtemp;
}

int main(int argc, char ** argv)
{
    int i,n;
    float a, *b, *c;
    if (argc < 3)
    {
	    fprintf(stderr, "Usage: %s size data_file\n", argv[0]);
	    return 1;
    }
    n = atoi(argv[1]);
    b = (float*) malloc(n * sizeof(float));
    c = (float*) malloc(n * sizeof(float));
    init_data_file(argv[2]);
    init_data_float(b,n);
    init_data_float(c,n);
    close_data_file();
    ddot_ur(n,b, c, &a);
    printf("%f\n",a);
    free(b);
    free(c);
}

