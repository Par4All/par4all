//Want to analyse the transformer/precondition 
//without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : if possible, have something like that P(i,j,k,p) {9i+10j==100, 1<=j, j<=10, ...}

#include<stdlib.h>

int main()
{
  int i, j, k, *p;
  i=0;
  j=1;
  
  if (rand()) {
    p = &i;
    k = i;
  } else {
    p = &j;
    k = j;
  }
  
  *p = 10;
  
  return 0;
}
