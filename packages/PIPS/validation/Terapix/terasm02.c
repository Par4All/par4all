#define N 512

void microcode(short image[N][N], short kernel[3][3], short new_image[N][N], short iter1, short iter2 )
{
    short i,j,k,l;
    for(i=0;i<iter1;i++)
    {
        for(j=0;j<iter2;j++)
        {
            new_image[i][j]=0;
            for(k=0;k<3;k++)
                for(l=0;l<3;l++)
                    new_image[i][j] = new_image[i][j] + image[i+k][j+l] + kernel[k][l];
            new_image[i][j]=new_image[i][j]/9;
        }
    }

}

void fake_copy_out(short image[N][N], short kernel[3][3],short new_image[N][N], short N1, short N2)
{
    microcode(image,kernel,new_image,N1,N2);
    for(N1=0;N1<N;N1++)
        printf("%d",new_image[0][N]);
}
