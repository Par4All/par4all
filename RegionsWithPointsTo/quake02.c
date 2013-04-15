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

void smvp(int nodes)
{
  int i,j;
  int **w2 = (int **) malloc(2 * sizeof(int *));
  for (j = 0; j < 2; j++) {
      for (i = 0; i < nodes; i++) {
      w2[j][i] = i;
    }
  }

  
  return;
}
