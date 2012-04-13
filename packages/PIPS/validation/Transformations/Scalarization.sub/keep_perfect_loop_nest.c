

int main() {
  int i,j;
  int N=100;
  int A[N][N];
  int u1[N];
  int u2[N];
  
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      A[i][j]=A[i][j]+u1[i]+u2[i];
  
}
