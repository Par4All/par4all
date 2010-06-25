int N;

short alphablend(int n,int m, short alpha,short a[n][m],short b[n][m], short c[n][m])
{
    short i,j;
here:    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            a[i][j]=b[i][j]*alpha +c[i][j]*(100-alpha);
}
