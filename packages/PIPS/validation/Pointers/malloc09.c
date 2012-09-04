/* Check detection of memory leaks due to a free of an object
 * containing a pointer.
 *
 * Issue: the source code was bugged 
 *
 * Bug: a non-constant source is generated, *HEAP*_l_10[0], instead of
 * *HEAP*_l_11
 *
 * Bug 2: the memory leak due to the free is not detected
 *
 * Bug 3: the numbering of context points-to stubs seems to be
 * erratic. It is probably not reseted each time a new module is
 * analyzed. FI: I do not know if Amira's naming scheme relies on this
 * counter to generate unique names across all modules.
 */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int *** ppp, i=1;

  ppp = (int ***) malloc(sizeof(int **));
  *ppp = (int **) malloc(sizeof(int *));
  **ppp = (int *) malloc(sizeof(int));
  ***ppp = i;

  free(ppp);

  return 0;
}
