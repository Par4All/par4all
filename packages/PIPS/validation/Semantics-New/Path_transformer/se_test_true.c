#include <stdio.h>
#include <stdlib.h>

int main()
{
  int i,k1=42,k2=42;
  sb:
    i=4;
  i=i+5;
  i = i*3;
  if(i>0){
    k1++;
    se:
      i = i+4;
  }
  else{
    k2++;
    i--;
  }
  return i;
}




  
