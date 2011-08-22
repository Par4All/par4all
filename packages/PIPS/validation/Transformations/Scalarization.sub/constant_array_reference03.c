// Special case of constant regions, here even a reference to a
// constant array element

// Excerpt of constant_array_reference01.c

#include <stdlib.h>
#include <stdio.h>

int main() {
 int n;
  // The declarations below are legal C for gcc, even when n is not
  // initialized by scanf()!
 scanf("%d", &n);
 int a[n], b[n];
 int i;

 a[0] = 2;
 // This loop use only one element of a, it should be scalarized
 // Since b is not used later, its value does not matter and it can be
 // scalarized too, however suprising this is
 for(i=0; i<n; i++) {
   b[i] = rand() * a[0]/10;
 }
 printf("%d\n", b[0]);
 return 0;
}
