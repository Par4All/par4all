#include <stdio.h>
#define N 512

void microcode(short (*image)[N], short kernel[1], short (*new_image)[N], short iter )
{
    short j;
    for(j=0;j<iter;j++)
        (*new_image)[j]=kernel[0]*(*image)[j];
}

void fake_copy_out(short image[N][N], short kernel[1],short new_image[N][N], short N1, short N2)
{
    microcode(&image[N1],kernel,&new_image[N1],N2);
    for(N2=0;N2<N;N2++)
        printf("%d",new_image[N1][N2]);
}
