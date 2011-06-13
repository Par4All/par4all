#include <stdio.h>

int a[1];
void kernel() {
  a[0]=1;
}

int main() {
  a[0] = 0;

  kernel();

  printf("%d\n",a[0]);
}
