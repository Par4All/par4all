// FI: the memory leak cause by the loop is not detected

#include <malloc.h>
#include <stdio.h>
int main(int argc, char *argv[])
{
  int *p, i, j;
  i = 0;
  j = 1;

 for(i = 1; i<5; i++){

  p = (int *) malloc(sizeof(int));

 }


 return 0;
}
