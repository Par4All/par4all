#include <stdio.h>
#include <stdlib.h>

/* what we should find: 
 {
 T(i,k1,k2,n) 
  - n#init + n#new  == 1 ,
  i#new >= 0
  i#new <= n#new - 1
  k1#new - k1#init  ==  i#new + 1 ,
  k2#new - k2#init  ==  i#new ,
 } 
*/

int main()
{
  int a[20], b[20];

  int i=0,j=0,k1=0, k2 =0, n=20;
  sb:
    b[i] = a[i]+a[i+1];
  n++;
  for(i=0;i< n;i++){
    k1 ++;
    se: 
      a[i] = a[i]+1;
    k2 ++;
  }
  return i;
}



  
