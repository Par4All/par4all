#include <stdio.h>


int main() {
   int a[10],b[10],c[10];

   for(int i=0; i<10; i++) {
     a[i] = 1;
   }

   for(int i=0; i<10; i++) {
     a[i] = 0;
     for(int j=i;j<10;j++) {
       // Scalarization of a[] make this affectation wrong !
       b[j] = a[j];
     }
   }

   for(int j=0;j<10;j++) {
     printf("%d - %d\n",b[j],a[j]);
   }

   return 0;
}
