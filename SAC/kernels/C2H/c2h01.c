void average(short* a, short* b, short *c, int n)
{
    int i;
    for(i=0;i<n;i++)
        a[i]=(b[i]/2) + (c[i]/2);
}
