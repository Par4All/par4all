#include <stdio.h>
#include <stdlib.h>


/*
  i#init >=0 is lost because of the use of the transitive closure
  i#new == n#init is approximated by i#new  >= n#init because of the use of transitive closure
*/

int main(int x)
{
  int a[20], b[20];

  int i=0,k1=0,k2=0,n;
  for(i=0;i< n;i++){
    k1++;
    sbegin:
      a[i] = a[i]+1;   
    k2++;
  }
  n++;
  send:
    b[i] = a[i]+a[i+1]; 
  return i;
}




  
