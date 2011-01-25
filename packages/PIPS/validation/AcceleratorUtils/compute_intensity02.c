#include <stdlib.h>
#include <stdio.h>

#define KFIR(k) void fir_##k##_(int n, int in[n], int out[n], int kernel[1024]) {\
    for (int i = 0; i < n - 1 + 1; ++i)\
    {\
        out[i] = 0.0;\
        for (int j = 0; j < k; ++j)\
            out[i] += in[i + j] * kernel[j];\
    }\
}
KFIR(1)
KFIR(2)
KFIR(4)
KFIR(8)
KFIR(16)
KFIR(32)
KFIR(64)
KFIR(128)
KFIR(256)
KFIR(512)
KFIR(1024)

    int
main (int argc, char *argv[])
{
    int n = argc == 1 ? 1000 : atoi (argv[1]);
    if (n > 1)
    {
        {
            int (*A)[n], (*B)[n], c = 0;
            int C[1024];
            A = (int (*)[n]) malloc (sizeof (int) * n);
            B = (int (*)[n]) malloc (sizeof (int) * n);
            for (int j = 0; j < n; j++)
                (*B)[j] = j;
            for (int l = 0; l < 1024; l++)
                C[l] = l;
            fir_1_ (n, *A, *B, C);
            fir_2_ (n, *B, *A, C);
            fir_4_ (n, *A, *B, C);
            fir_8_ (n, *B, *A, C);
            fir_16_ (n, *A, *B, C);
            fir_32_ (n, *B, *A, C);
            fir_64_ (n, *A, *B, C);
            fir_128_ (n, *B, *A, C);
            fir_256_ (n, *A, *B, C);
            fir_512_ (n, *B, *A, C);
            fir_1024_ (n, *A, *B, C);
            for (int i = 0; i < n; i++)
                c += (*B)[i];
            printf ("%f\n", c);
            free (A);
            free (B);
        }
    }
    return 0;
}
