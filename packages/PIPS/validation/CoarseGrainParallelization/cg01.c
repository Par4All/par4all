#define N 64
float imagein_re[N][N];
float imagein_im[N][N];
int main(int argc, char *argv[]) {
  int i, j;

  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      imagein_re[i][j]=2*i;
    }


  return 0;
}
