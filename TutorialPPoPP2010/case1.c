#include <stdio.h>

#define image_size 10

void func1(int n, int m,float a[n][m], float b[n][m], float h)
{
  float x;
  int i,j;
  for(i = 1; i<=n; i++)
    for(j = 1; j<=m; j++){
    x = i*h + j;
    a[i][j] = b[i][j]*x;
  }
}

int main()
{
  float a[image_size][image_size],b[image_size][image_size],h;
  int i,j ;
  for(i = 1; i<=image_size; i++)
    for(j = 1; j<=image_size; j++)
      b[i][j] = 1.0;
  h=2.0;
  func1(image_size,image_size,a,b,h);
  for(i = 1; i<=image_size; i++)
    for(j = 1; j<=image_size; j++)
      fprintf(stderr, "a[%d] = %f \n",i,a[i][j]);
}
