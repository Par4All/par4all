#include "stdio.h"

typedef float t_real;

// real, dimension(n1,n2,n3) :: u, v
// real, dimension(-L:L) :: c

//  parameter(L=4, n1=100, n2=100, n3=100)

void stencil8 (int L, int n1, int n2, int n3, t_real u[n1][n2][n3], t_real v[n1][n2][n3], t_real c[2L+1],
	       int is1, int ie1, int is2, int ie2,
	       int is3, int ie3) {
  // Stencil length : 2*L

  int i1,i2,i3;
  t_real c_4,c_3,c_1, c_2, c0, c1, c2,c3,c4;

  c_4 = c[L-4]; c_3 = c[L-3]; c_2 = c[L-2]; c_1 = c[L-1];
  c0 = c[L];
  c4 = c[L+4]; c3 = c[L+3]; c2 = c[L+2]; c1 = c[L+1];

  //do i1=is1+L,ie1-L
  for (i1=is1+L; i1<ie1-L ; i1++) {
    //do i2=is2+L,ie2-L
    for (i2=is2+L; i2<ie2-L ; i2++) {
      //  do i3=is3+L,ie3-L
      for (i3=is3+L; i3<ie3-L ; i3++) {
	u[i1][i2][i3]=
	    c_4 * (v[i1-4][i2][i3] + v[i1][i2-4][i3] + v[i1][i2][i3-4])
	  + c_3 * (v[i1-3][i2][i3] + v[i1][i2-3][i3] + v[i1][i2][i3-3])
	  + c_2 * (v[i1-2][i2][i3] + v[i1][i2-2][i3] + v[i1][i2][i3-2])
	  + c_1 * (v[i1-1][i2][i3] + v[i1][i2-1][i3] + v[i1][i2][i3-1])
	  + c0  *  v[i1][  i2][i3] * 3.f
	  + c1  * (v[i1+1][i2][i3] + v[i1][i2+1][i3] + v[i1][i2][i3+1])
	  + c2  * (v[i1+2][i2][i3] + v[i1][i2+2][i3] + v[i1][i2][i3+2])
	  + c3  * (v[i1+3][i2][i3] + v[i1][i2+3][i3] + v[i1][i2][i3+3])
	  + c4  * (v[i1+4][i2][i3] + v[i1][i2+4][i3] + v[i1][i2][i3+4]);
      }
    }
  }
}

// initialize the array, with the give value
void init ( int n1, int n2, int n3,t_real u[n1][n2][n3], t_real val) {
  int i = 0, j = 0, k = 0;
  for (i=0; i<n1 ; i++) {
    for (j=0; j<n2 ; j++) {
      for (k=0; k<n3 ; k++) {
	u[i][j][k] = val;
      }
    }
  }
  return;
}

// sum all the elements of the array
t_real sum  ( int n1, int n2, int n3,t_real u[n1][n2][n3]) {
  t_real result = 0;
  int i = 0, j = 0, k = 0;
  for (i=0; i<n1 ; i++) {
    for (j=0; j<n2 ; j++) {
      for (k=0; k<n3 ; k++) {
	result += u[i][j][k];
      }
    }
  }
  return result;
}

int main (int argc, char * argv[]) {

  int is1,ie1,is2,ie2,is3,ie3,i;
    int L = 4;
    int n1 = 100;
    int n2 = 100;
    int n3 = 100;
    if(argc >100000) n1=n2=n3=L=78;
    {
  t_real v[n1][n2][n3];
  t_real u[n1][n2][n3];
  t_real c[2*L+1];
  is1=0;ie1=n1;
  is2=0;ie2=n2;
  is3=0;ie3=n3;

  for (i=0; i<2*L+1; i++) {
    c[i] = 3.0f;
  }

  // Simple case
  init (n1,n2,n3,u , 1.0f);
  init (n1,n2,n3,v , 1.0f);
  stencil8(L,n1,n2,n3,u,v,c,is1,ie1,is2,ie2,is3,ie3);

  printf ("the sum is : %f\n", sum (n1,n2,n3,u));
    } return 0;
}
