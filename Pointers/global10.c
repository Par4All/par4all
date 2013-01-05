/* Building points-to in demand for global variables
 *
 * See what happens when the callee builds up the global stubs
 */

int ***p;

int global10()
{
  int **q, *r, i;

  q = *p;
  *q = &i;
  r = &**q;
  return *r;
}

int main()
{
  int i = global10();
  return i;
}
