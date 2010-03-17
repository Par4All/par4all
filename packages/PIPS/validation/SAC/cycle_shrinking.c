#define n 100
#define k 4
void shrink(float a[n+k], float b[n+k], float c[n+k])
{
    int i;
    // icc cannot vectorize this
    for(i=0;i<n;i++)
    {
        a[i+k] = b[i];
        b[i+k]=a[i]+c[i];
    }
}

