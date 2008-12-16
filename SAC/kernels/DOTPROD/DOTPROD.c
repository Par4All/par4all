void dotprod(int* a, int *b, int *c)
{
    int i;
    for(i=0;i<1000000;++i)
    {
        a[i] = b[i] + c[i];
    }


    return;
}


