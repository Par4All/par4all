/* Parallelization with pointers and linearized array
 *
 * Plus bug in initialization of a.array in points-to pass
 */

#include<stdlib.h>
#include<stdio.h>

typedef struct {
  int d1;
  int d2;
  double *array;
} array_2D; 

void pointer12(array_2D a)
{
  int i, j;
  a.d1 = 5;
  a.d2 = 4;
  a.array = malloc((a.d1*a.d2)*sizeof(double)); 
  for(i=0; i<a.d1; i++)
    for(j=0; j<a.d2; j++)
      a.array[a.d2*i+j] = 0.;
 
}

