//Want to analyse the transformer/precondition 
// for a if in lhs and rhs and deref lhs and rhs without a temp
// without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : (i, j) and (l1, l2) have the same info


#include<stdlib.h>

int main()
{
  int i, j, l1, l2, m, n, *p, *q;
  i=0; l1=i;
  j=1; l2=j;
  m=10;
  n =11;
  
  if (rand()) {
    q = &m;
  } else {
    q = &n;
  }
  
  if (rand()) {
    p = &i;
    l1 = rand()?m:n;
    l1 = l1;
  } else {
    p = &j;
    l2 = rand()?m:n;
    l2 = l2;
  }
  
  *p = *q;
  
  return 0;
}
