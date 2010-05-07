void foo_l(int a[4],int b[40])
{
    int i;
    for(i=0;i<10;i++)
    {
        a[0]=a[0]+b[i*4];
        a[1]=a[1]+b[1+i*4];
        a[2]=a[2]+b[2+i*4];
        a[3]=a[3]+b[3+i*4];
    }
}
