// dereferencing an unitialized pointer yields a user error
#include <stdio.h>
int main()
{
  int **a;
  int b = 1;
  a[0] = &b; 
  printf("\nb= %d, a[0] = %d\n", b, a[0][0]);
  return(0);
}
