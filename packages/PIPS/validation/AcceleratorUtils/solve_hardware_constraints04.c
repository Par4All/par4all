int N;
void doiv(int a[128][128])
{
    int i,j;
here:    for(j=0;j<2*N;j++)
    {
        for(i=0;i<N;i++)
        a[i][j]=1;
    }
}
