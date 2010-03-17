#define N 100
#define M 4
void filtre(float x[N], float y[N], float h[M])
{
    int i,k;
    for(i=M-1;i<N;i++)
loop:        for(k=0;k<M;k++)
            y[i]=y[i]+x[i-k]*h[k];
}
