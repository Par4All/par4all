/* Check that memory leaks and indirect memory leaks created by an
 * assignment are detected: struct case
 *
 * Same as malloc18.c, but the memory leaks are created by a free()
 * instead of an ssignment.
 *
 * Bug: some buckets based on *HEAP*_l_17 are not removed from the
 * points-to relation. This is not a leak undetected. It's about
 * missing element in a kill set.
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

  ps->first = (int *) malloc(10*sizeof(int));
  ps->second = (int *) malloc(10*sizeof(int));

  free(ps);

  return 0;
}
