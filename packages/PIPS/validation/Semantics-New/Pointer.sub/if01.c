//Want to analyse the transformer/precondition 
// for a if in lhs and deref lhs
// without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : (i, j) and (l1, l2) have the same info
//      if possible, have something like that P(i,j,p, ...) {9i+10j==100, 1<=j, j<=10, ...}

#include<stdlib.h>

int main()
{
  int i, j, l1, l2, *p;
  i=0; l1=i;
  j=1; l2=j;
  
  if (rand()) {
    p = &i;
    l1 = 10;
  } else {
    p = &j;
    l2 = 10;
  }
  
  *p = 10;
  
  return 0;
}
