// same result with if04 except for the info on k, why??
//Want to analyse the transformer/precondition 
// for a if in lhs and rhs and deref lhs and rhs without a temp
// without any points-to information : lost every information
// with effect_with_points_to info : lost the value of i and j
// with effect_with_points_to and points-to info : (i, j) and (l1, l2) have the same info


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
    if (rand()) {
      q = &m;
      k = m;
    } else {
      q = &n;
      k = n;
    }
    p = &i;
    l1 = k;
    l1 = l1;
  } else {
    if (rand()) {
      q = &m;
      k = m;
    } else {
      q = &n;
      k = n;
    }
    p = &j;
    l2 = k;
    l2 = l2;
  }
  
  //We lost the info on k why? 10<=k<=11
  *p = *q;
  
  return 0;
}
