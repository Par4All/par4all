
#include<stdlib.h>

int main()
{
  int k, m, n, l1, l2;
  l1=0;
  l2=1;
  m=10;
  n =11;
  k = -1; 
  
  if (rand()) {
    if (rand()) {
      k = m;
    } else {
      k = n;
    }
    l1 = k;
  } else {
    if (rand()) {
      k = m;
    } else {
      k = n;
    }
    l2 = k;
  }
  
  //We lost the info on k why? useless? 10<=k<=11
  
  return 0;
}
