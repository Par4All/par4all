//Want to analyse the transformer/precondition 
// for a if in lhs and rhs and deref lhs and rhs without a temp
// without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : (i, j) and (l1, l2) have the same info


#include<stdlib.h>

void inc01(int *p)
{
  (*p)++;
}
int main()
{
  int i = 0;
  inc01(&i);
  inc01(&i);
  
  return i;
}
