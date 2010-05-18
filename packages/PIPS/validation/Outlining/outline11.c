#include <stdio.h>

int main(int argc, char *argv[]) {
  int i, j;
  int a[10],b[10][10];

  for(i=0;i<10;i++)
 init_kernel:
     a[i]=b[i][i]=0;

  for(i=0;i<10;i++)
 compute_kernel:
     a[i]+=b[i][0];

  printf("Value is %d\n", j);

  return 0;
}
