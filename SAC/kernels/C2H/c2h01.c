void average(short* a, short* b, short *c)
{
    int i;
    for(i=0;i<100;i++)
    {
        a[i]=(b[i]/2) + (c[i]/2);
//        a[i]+=(b[i]/2) + (c[i]/2);
    }
}
