/* Check that memory leaks created by an assignment are detected */

#include <stdlib.h>

int main()
{
  int **p = (int **) malloc(10*sizeof(int *));

  *p = (int *) malloc(10*sizeof(int));

  p = NULL;

  return 0;
}
