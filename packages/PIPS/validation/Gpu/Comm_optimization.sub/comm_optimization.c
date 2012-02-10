#include <stdio.h>
extern int atoi(const char *nptr);

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

  display(n,c);

  kernel_add(n,a,b,c);


  display(n,a);


}
