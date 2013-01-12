/* Check that possible aliases and memory leaks are not overestimated
 *
 * Interprocedural test of main
 *
 * p and q points towards the same abstract heap location
 *
 * Bug: the fact that any heap element is an abstract location
 * representing several memory buckets is ignored and q is assumed
 * undefined when it is not. It may be undefined but it also may point
 * toward the heap.
 */

#include <malloc.h>

int * foo(void)
{
  int *p = (int *) malloc(sizeof(int));
  return p;
}

int main()
{
  int i=1, *p=foo(), *q=foo();

  free(p);

  return i;
}
