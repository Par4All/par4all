#include <stdio.h>
#define N 100
#define M 4
void filtre(float x[N], float y[N], float h[M])
{
    int i,k;
    for(i=M-1;i<N;i++)
        for(k=0;k<M;k++)
            y[i]=y[i]+x[i-k]*h[k];
}

main()
{
    int i;
    float x[N],y[N],h[M];
    for(i=0;i<N;i++)
        x[i]=y[i]=i;
    for(i=0;i<M;i++)
        h[i]=i;
    filtre(x,y,h);
    for(i=0;i<N;i++)
        printf("%f-",y[i]);
    return 0;
}
