#include <stdio.h>
#include <stdlib.h>

static void mat_mul(int l, int m, int n,
		    double C[l][n],
		    double A[l][m],
		    double B[m][n])

{
  for(int i=0; i<l; i++)
    for(int j=0; j<n; j++) {
      C[i][j] = 0.0;
      for(int k=0; k<m; k++)
	C[i][j] += A[i][k]*B[k][j];
    }
}

static void mat_mul_const(int m, int n,
			  double B[m][n],
			  double A[m][n],
			  double d)

{
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      B[i][j] = A[i][j] * d;
}


static void mat_add(int m, int n,
		    double C[m][n],
		    double A[m][n],
		    double B[m][n])

{
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      C[i][j] = A[i][j] + B[i][j];
}

static void mat_zero(int m, int n, 
		     double A[m][n])
{
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      A[i][j] = 0.0;
}

static double minim(double a,
		    double b)
{
  if(a<b) return a;
  return b;
}

static double maxim(double a,
		    double b)
{
  if(a>b) return a;
  return b;
}

static void skip(void){
  return;
}

static void send(double u, int a){
  printf("result = %f, %d\n", u, a);
}

static void receive(double * d, int n)
{
  *d = rand();
}


int feron() {
  double Ac[2][2] = { {0.4990, -0.0500}, {0.0100, 1.0000} };
  double Bc[2][1] = { {1.0}, {0.0} };
  double Cc[1][2] = { {564.48, 0.0} };
  double Dc = -1280.0;
  double xc[2][1];
  mat_zero(2, 1, xc);
  double y, yd;
  receive(&y, 2);
  receive(&yd, 3);
  double yc = y - yd, u;
  while(1) {
    yc = maxim(-1.0, minim(y-yd, 1.0));
    skip();
    double tmp0[1][1];
    mat_mul(1, 2, 1, tmp0, Cc, xc);
    u = tmp0[0][0] + Dc*yc;
    double tmp1[2][1], tmp2[2][1];
    mat_mul(2, 2, 1, tmp1, Ac, xc);
    mat_mul_const(2, 1,tmp2, Bc, yc);
    mat_add(2, 1, xc, tmp1, tmp2);
    send(u, 1);
    receive(&y, 2);
    receive(&yd, 3);
    skip();
  }
  return 0;
}
