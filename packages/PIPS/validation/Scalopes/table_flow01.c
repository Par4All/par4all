/*Show that KERNEL_LOAD_STORE handles only table that size is known*/
#include <stdlib.h>
int main(){
  int i;
  int * a;
  int b[10] = {0};

  a = (int *) malloc(10*sizeof(int));
 #pragma scmp task
  for(i=0;i<10;i++){
    b[i]=i;
    a[i]=i;
  }

  a[0]++;
  b[0]++;
  return 0;
}
