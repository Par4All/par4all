//Want to analyse the transformer/precondition 
// for a if in lhs and rhs and deref lhs and rhs without a temp
// without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : (i, j) and (l1, l2) have near the same info


#include<stdlib.h>

int main()
{
  int i, j, k, l1, l2, m, n, *p, *q;
  i=0; l1=i;
  j=1; l2=j;
  m=10;
  n =11;
  //l1=-1; l2 =-1;
  
  if (rand()) {
    q = &m;
    k = m;
  } else {
    q = &n;
    k = n;
  }
  
  
  
  if (rand()) {
    p = &i;
    l1 = k;
  } else {
    p = &j;
    l2 = k;
  }
  
  *p = *q;
  
  return 0;
}
