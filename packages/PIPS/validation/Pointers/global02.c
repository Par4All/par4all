/* Building points-to in demand for global variables
 *
 * The indexing of the stub depends on the property setproperty
 * POINTS_TO_STRICT_POINTER_TYPES.
 */

int **p;

int main()
{
  int **q, *r, i;

  q = p;
  *q = &i;
  r = &**q;

  return 0;
}
