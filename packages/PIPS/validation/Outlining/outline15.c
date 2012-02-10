#include <stddef.h>
typedef int fc_int; /* you know of this famous data type ? */
typedef fc_int _int;

/* this is a classical convolution */
void outline15( _int W, int H, int R,
			    float Dst [W][H], float Src [W][H], float Kernel[R] )
{
    size_t y,x,k;
    float sum = 0;
    for(y = 0; y < H; y++)
        for(x = 0; x < W; x++){
            sum = 0;
here:for(k = -R; k <= R; k++){
                struct { int _; } _ = { 0 };
                int d = y + k + _._;
                if(d >= 0 && d < H)
                    sum += Src[y][x] * Kernel[R - k];
            }
            Dst[y][x] = sum;
        }
}
