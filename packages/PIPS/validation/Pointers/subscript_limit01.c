#include<stdlib.h>

int main() {
  int i;
  int *p = (int*) malloc(10*sizeof(int));
  for( i=0 ; i<10 ;i++ )
    *p++ = 0;

 return 0;
}
