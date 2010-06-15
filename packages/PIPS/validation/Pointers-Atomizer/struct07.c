#define N 256
#define M 256

typedef struct {
  float re;
  float im;
} complex;

complex imagein[N][N], imageout[M][M];


void struct07()
{
  extern complex imagein[N][N], imageout[M][M];
  int i, j, k, l;
  float z1, z2;
  float x[N][N];

  for(k=0;k<N;k++) {
    imageout[k][1].re = z1;
  }

  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      x[i][j] = 0.;
}
