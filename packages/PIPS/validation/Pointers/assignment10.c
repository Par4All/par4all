/* Check assigment of arrays
 *
 */

int assignment10()
{
  struct {
    int a[10];
  } s;
  int *q=s.a;

  return *q;
}
