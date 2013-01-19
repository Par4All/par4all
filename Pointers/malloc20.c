/* Check that memory leaks and indirect memory leaks created by an
 * assignment are detected: struct case
 *
 * Same as malloc18.c, but pointer "qs" is added to keep track of all
 * allocated buckets.
 */

#include <stdlib.h>

int main()
{
  struct s {
    int * first;
    int * second;
  };
  typedef struct s s_t;
  s_t *ps = (s_t *) malloc(10*sizeof(s_t));
  s_t * qs = ps + 1;
  int * fp;

  ps->first = (int *) malloc(10*sizeof(int));
  ps->second = (int *) malloc(10*sizeof(int));

  ps = NULL;

  fp = (qs-1)->first;
  *fp = 0;

  return *fp;
}
