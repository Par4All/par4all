#include <stdio.h>

void kernel(int n, int a[n]) {
  for(int j=0; j<n; j++) {
    a[j] = 0;
  }
}

int main(int argc, char **argv) {
 int i,j;
 int n = argv;
 int a[n]; // Because of the C99 declaration, we are not precise enough ! (see loop04_static.c for a C89 version)
 int sum;
 n = n - 1;
 for(i=0; i<n; i++) {
    a[9]=a[9]+1;
    kernel(n,a);
 }

 printf("%d\n",a[9]);
}


