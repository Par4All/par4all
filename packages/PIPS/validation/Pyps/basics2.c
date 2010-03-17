void do_convol(int i, int j, int n, int a[n][n],int b[n][n], int kernel[3][3])
{
    int k,l;
    b[i][j]=0;
    for(k=0;k<3;k++)
    for(l=0;l<3;l++)
        b[i][j]+=a[i+k-1][j+l-1]*kernel[k][l];

}
void convol(int n,int a[n][n],int b[n][n], int kernel[3][3])
{
    int i,j;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            do_convol(i,j,n,a,b,kernel);
}
