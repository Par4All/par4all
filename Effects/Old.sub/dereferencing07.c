#include <stdlib.h>
#include <stdio.h>

#define N 5


void foo(int *col)
{
  *col++;
  
  return;
}

int main() 
{
  int x, tab1[10], tab2[10], *p;
  x++;
  x += 3;
  *(p+tab1[tab2[x]]) = 0;
return 1;
}
