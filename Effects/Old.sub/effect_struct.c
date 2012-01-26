enum { N = 9 };

typedef struct	{
		float re;
		float im;
		} complex;

complex imagein[N][N];

float
main(int argc, char *argv[]) {
  int i,j,k;

  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      imagein[i][j].re=0.0;
      imagein[i][j].im=0.0;
    }
  return imagein[2][1].re;
}
