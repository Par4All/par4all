
/* this is a classical convolution */
void outline14( int W, int H, int R,
			    float Dst [W][H], float Src [W][H], float Kernel[R] )
{
    int y,x,k;
    float sum = 0;
    for(y = 0; y < H; y++)
        for(x = 0; x < W; x++){
            sum = 0;
here:for(k = -R; k <= R; k++){
                int d = y + k;
                if(d >= 0 && d < H)
                    sum += Src[y][x] * Kernel[R - k];
            }
            Dst[y][x] = sum;
        }
}
