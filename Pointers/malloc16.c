/* Check that memory leaks created by an assignment are detected */

#include <malloc.h>

int main()
{
  int *p = (int *) malloc(10*sizeof(int)), i;

  p = &i;

  return 0;
}
