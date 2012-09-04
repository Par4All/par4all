#include<stdlib.h>
int main()
{
  int *pi, *qi;

  pi = (int *) malloc(sizeof(int));
  S: qi = pi; 
  free(pi);

  return 0;
}
