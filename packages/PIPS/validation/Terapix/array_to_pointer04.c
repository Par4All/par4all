typedef unsigned int size_t;
void vector_product(size_t n, double a[n][n],double b[n],double c[n])
{
    size_t i,j;
    double res = 0.;
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
            c[i]+=a[i][j]*b[j];
}
