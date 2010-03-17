#define N 64
#define M 256
float imagein_re[N][N], imageout_re[M][M];
float imagein_im[N][N], imageout_im[M][M];

void beatrice01() {
  int i,j;

  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      imagein_re[i][j]=0.0;
      imagein_im[i][j]=0.0;
    }

 outline_compute:
  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      imagein_re[i][j]= 1 + imagein_im[i][j];
    }
}
