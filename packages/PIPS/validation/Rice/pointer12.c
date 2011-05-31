/* Parallelizarion with pointers */
#include<stdlib.h>
 typedef struct{
    int d1;
    int d2;
    double *array;
  }array_2D; 

void pointer12(array_2D a)
{
  int i, j;
  
  for(i=0; i<a.d1; i++)
    for(j=0; j<a.d2; j++)
      a.array[a.d2*i+j] = 0.;
}
