/* This example is from  /MYSTEP/CodeExamples/FSC_benchs/SPEC/320.equake_m/src/quake.c */
/* pragma are deleted */

/*--------------------------------------------------------------------------*/ 
/* Matrix vector product - basic version                                    */

#include<stdlib.h>


typedef struct smallarray_s {
  double first;
  double second;
  double third;
  double pad;
} smallarray_t;

void smvp(int nodes, double (*A)[3][3], int *Acol, int *Aindex,
	  double (*v)[3], double (*w)[3])
{
  int i,j;
  int Anext, Alast, col;
  double sum0, sum1, sum2;
  smallarray_t **w1 = (smallarray_t **) malloc(2* sizeof(smallarray_t *));
  int **w2 = (int **) malloc(2 * sizeof(int *));
  for (j = 0; j < 2; j++) {
    for (i = 0; i < nodes; i++) {
      w2[j][i] = 0;
    }
  }

  for (i = 0; i < nodes; i++) {
    Anext = Aindex[i];
    Alast = Aindex[i + 1];

    sum0 = A[Anext][0][0]*v[i][0] + A[Anext][0][1]*v[i][1] + A[Anext][0][2]*v[i][2];
    sum1 = A[Anext][1][0]*v[i][0] + A[Anext][1][1]*v[i][1] + A[Anext][1][2]*v[i][2];
    sum2 = A[Anext][2][0]*v[i][0] + A[Anext][2][1]*v[i][1] + A[Anext][2][2]*v[i][2];

    Anext++;
    while (Anext < Alast) {
      col = Acol[Anext];

      sum0 += A[Anext][0][0]*v[col][0] + A[Anext][0][1]*v[col][1] + A[Anext][0][2]*v[col][2];
      sum1 += A[Anext][1][0]*v[col][0] + A[Anext][1][1]*v[col][1] + A[Anext][1][2]*v[col][2];
      sum2 += A[Anext][2][0]*v[col][0] + A[Anext][2][1]*v[col][1] + A[Anext][2][2]*v[col][2];

      if (w2[0][col] == 0) {
	w2[0][col] = 1;
	w1[0][col].first = 0.0;
	w1[0][col].second = 0.0;
	w1[0][col].third = 0.0;
      }
      
      w1[0][col].first += A[Anext][0][0]*v[i][0] + A[Anext][1][0]*v[i][1] + A[Anext][2][0]*v[i][2];
      w1[0][col].second += A[Anext][0][1]*v[i][0] + A[Anext][1][1]*v[i][1] + A[Anext][2][1]*v[i][2];
      w1[0][col].third += A[Anext][0][2]*v[i][0] + A[Anext][1][2]*v[i][1] + A[Anext][2][2]*v[i][2];
      Anext++;
    }

    if (w2[0][i] == 0) {
      w2[0][i] = 1;
      w1[0][i].first = 0.0;
      w1[0][i].second = 0.0;
      w1[0][i].third = 0.0;
    }

    w1[0][i].first += sum0;
    w1[0][i].second += sum1;
    w1[0][i].third += sum2;
  }



  for (i = 0; i < nodes; i++) {
    for (j = 0; j < 2; j++) {
      if (w2[j][i]) {
	w[i][0] += w1[j][i].first;
	w[i][1] += w1[j][i].second;
	w[i][2] += w1[j][i].third;
      }
    }
  }
}
