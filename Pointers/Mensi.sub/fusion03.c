#define N 100
int fusion03(int *D, int *E, int *F)
{
  int i, j;
for(i=1;i<=N;i++)
  D[i]=E[i]+F[i];

for(j=1;j<=N;j++)
  E[j]=D[j]*F[j];

 return 0;
}
