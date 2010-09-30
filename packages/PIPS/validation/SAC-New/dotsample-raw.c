#include <stdio.h>
#include <sys/time.h>

short dot_product(int size, short a[size], short b[size])
{

  int i;
  int sum=0;
  for(i=0; i<size; i++)
  {
    sum += a[i]*b[i];
  }
  return sum;
}

int main(int argc, char *argv[])
{
  int size = argc==1?32:atoi(argv[1]);
  short a[size], b[size];
  int i;
  short product;
  struct timeval stop,start;

  for(i=0; i<size; i++)

  {
    a[i]=i;
    b[i]=i;
  }

  gettimeofday(&start,0);
  product =dot_product(size,a,b);
  gettimeofday(&stop,0);
  printf("%d:%ld\n",product,(stop.tv_sec-start.tv_sec)*1000000+(stop.tv_usec-start.tv_usec));
  return 0;
}

