#include <stdio.h>
#include <stdlib.h>

/* what we should find: 
 {
 T(i,k,n) 
  - n#init + n#new  == 1 ,
  i#new == n#init,
  k#new - k#init  ==  n#init ,
 } 
*/

int main()
{
  int a[20], b[20];

  int i=0,j=0,k=0,n=20;
  sbegin:
    a[i] = a[i]+1;
  for(i=0;i< n;i++){
    k ++;
  }
  n++;
  send:
    b[i] = a[i]+a[i+1];
  return n;
}




