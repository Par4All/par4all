double get(double f[SIZE],int i)
{
    return f[i];
}
void foo(double A[SIZE], double B[SIZE][SIZE]);
int main(int argc, char **argv)
{
    int i,j;
    double a[SIZE],b[SIZE][SIZE];
    double s=0;
    for(i=0;i<SIZE;i++)
    {
        a[i]=rand();
        for(j=0;j<SIZE;j++)
            b[i][j]=rand();
    }
#ifdef TEST
#ifndef rdtscll
#define rdtscll(val) \
         __asm__ __volatile__("rdtsc" : "=A" (val))
#endif
    long long stop,start;
    rdtscll(start);
    foo(a,b);
    rdtscll(stop);
    printf("%lld\n",stop-start);
#else
    foo(a,b);
#endif
    for(i=0;i<SIZE;i++)
        for(j=0;j<SIZE;j++)
            s+=a[i]+b[i][j];
    return (int)s;
}

#ifdef TEST
int MAX0(int a,int b) { return ((a)>(b))?(a):(b); }
#endif

