int N;
void doiv(int a[128])
{
    int i;
here:    for(i=0;i<N;i++)
    {
        a[i]=a[i+1];
    }
}
