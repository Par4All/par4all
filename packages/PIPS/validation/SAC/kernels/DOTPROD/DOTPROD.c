#define N 200
float dotprod(float b[N], float c[N])
{
    int i;
    float a=0;
    for(i=0;i<1000000;++i)
    {
        a += b[i] * c[i] ;
    }
    return a;
}

int main()
{
    float a,b[N],c[N];
    int i;
    for(i=0;i<N;i++)
    {
        b[i]=i;
        c[i]=N-i;
    }
    a=dotprod(b,c);
    printf("%f",a);
    return 0;
}

