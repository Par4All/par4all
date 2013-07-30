//Want to analyse the transformer/precondition 
// for a if in rhs and deref rhs
// without any points-to information : lost the value of m
// with effect_with_points_to info : lost the value of m
// with effect_with_points_to and points-to info : l1 and l2 have the same info
//      have l1=i || l1=j (0<=l1<=1)

#include<stdlib.h>

int main()
{
  int k, l1, l2, m, n, *q;
  m=10;
  n =11;
  l1=-1; l2 =-1;
  
  
  if (rand()) {
    q = &m;
    k = m;
  } else {
    q = &n;
    k = n;
  }
  
  l1 = *q;
  l2 = k;
  
  return 0;
}
