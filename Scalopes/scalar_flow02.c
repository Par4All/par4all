/*
* Test that scalar are not handled by kernel_load_store
* if KERNEL_LOAD_STORE_SCALAR is false
*/
#include <stdio.h>

int main(){
  int i;
  int b;
  int c[10]={0};
  int a=2;

  a=0;
  b=0;

  #pragma scmp task
  for(i=0;i<10;i++){
    a++;
    b++;
    c[i]++;
  }

 #pragma scmp task
  for(i=0;i<5;i++){
    a--;
    b--;
  }

  return 0;
}
