#include <stdio.h>
#include <stdlib.h>

// write a && b
void init(int n, int a[n], int b[n]) {
  for(int i=0; i<n;i++) {
    a[i] = i;
    b[i] = i;
  } 
}

// c = a + b
void kernel_add(int n, int c[n], int a[n], int b[n]) {
  for(int i=0; i<n;i++) {
    c[i] = a[i]+b[i];
  } 
}

// c = a + b
void wrap_kernel_add(int n, int c[n], int a[n], int b[n]) {
  kernel_add(n, c[n], a[n], b[n]);
}


// Read a
void display(int n, int a[n]) {
  for(int i=0; i<n;i++) {
    printf("%d ",a[i]);
  } 
  printf("\n");
}

int main(int argc, char **argv) {
  int n = atoi(argv[1]);
  int a[n], b[n], c[n];

  init(n,a,b);


  kernel_add(n,c,a,b);

  // This wrapper contains a kernel call
  // We use to kill a def on the gpu by a use here
  // this is because when we get the propers effects 
  // for this call we can't know if it a use in a 
  // kernel or on the CPU
  wrap_kernel_add(n,a,b,c);

  display(n,c);


}
