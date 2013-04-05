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
    //    for (i = 0; i < nodes; i++) {
      w2[j] = &i;
      //}
  }

  
  return;
}
