/* Check assigment of arrays to pointers
 *
 * Bug: r_cell_reference_to_type()...
 */

int assignment11()
{
  struct {
    int a[10][10];
  } s;
  int *q=s.a[0];

  return *q;
}
