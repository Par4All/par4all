        
#include <stdlib.h>
#include <stdbool.h>


/*
 *  a00 * a[idx-n-1]     a10 * a[idx-1]      a20 * a[idx+n-1]
 *  a01 * a[idx-n]       a11 * a[idx]        a21 * a[idx+n]
 *  a02 * a[idx-n+1]     a12 * a[idx+1]      a22 * a[idx+n+1]
 */

void conv_cpu(float *a, float *c, int n,
              float a00, float a10, float a20,
              float a01, float a11, float a21,
              float a02, float a12, float a22)
{
    int i, j;
    for (i=0; i<n; ++i) {
        for (j=0; j<n; ++j) {
            int idx = i * n + j;

            bool right = i > 0;
            bool left = i < n-1;
            bool top = j > 0;
            bool bottom = j < n-1;

            c[idx] = ((right & top) ? a00 * a[idx-n-1] : 0)
                + (right ? a10 * a[idx-1] : 0)
                + ((right & bottom) ? a20 * a[idx+n-1] : 0)
                + (top ? a01 * a[idx-n] : 0)

                + a11 * a[idx]
                + (bottom ? a21 * a[idx+n] : 0)
                + ((left & top) ? a02 * a[idx-n+1] : 0)
                + (left ? a12 * a[idx+1] : 0)
                + ((left & bottom) ? a22 * a[idx+n+1] : 0);
        }
    }
}

int main(int argc, char *argv[])
{
    float * in, *out;
    float convol[3][3] = { {0.1,0.2,0.3},{0.1,0.2,0.3},{0.1,0.2,0.3}};
    int i,n;
    n=atoi(argv[1]);
    in=malloc(sizeof(float)*n);
    out=malloc(sizeof(float)*n);
    conv_cpu(in, out, n, convol[0][0], convol[0][1] , convol[0][2],
             convol[1][0], convol[1][1] , convol[1][2],
              convol[2][0], convol[2][1] , convol[2][2]);
    for(i=0;i<n;i++) printf("%f\n",out[i]);
    return 0;
}

