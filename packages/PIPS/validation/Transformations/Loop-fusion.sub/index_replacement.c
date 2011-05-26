#include <stdio.h>


int main() {
   int n = 10;
   int a[n];
   int i,j;

   
  for(j = 0; j < n; j += 1)
    a[j]=j;

  for(i = 0; i < n; i += 1)
    // Here we use the indice of first loop, it has to be renamed if we fuse !
    a[i]=j;

  for(i = 0; i < n; i += 1)
    printf("%d - ",a[i]);

   printf("\n");
   return 0;
}

