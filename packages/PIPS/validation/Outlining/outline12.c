#include<stdio.h>
#define N 64
float imagein_re[N][N];
float imagein_im[N][N];
int main(int argc, char *argv[]) {
  int i, j;

  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
 init_kernel:
      imagein_re[i][j]=2*i;
      imagein_im[i][j]=3*j;
    }

  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
 compute_kernel:
      imagein_re[i][j]= 1 + imagein_im[i][j];
    }

  for(i=0;i<N;i++) {
    printf("Line %d:\n");
    for(j=0;j<N;j++) {
      printf("%d ", imagein_re[i][j]);
    }
    puts("\n");
  }

  return 0;
}
