/* Pointer in union, expanded version of union02
 *
 * The pointer in union x is overwritten by the assignment to another
 * union member, so *(px->psecond) is indeterminate.
 */

int union03()
{
  union one {
    int first;
    int * psecond;
  } x, *px;
  int y[10];

  px = &x;
  px->psecond = &y[0];
  //px->first = 1;
  // The dereferenced pointer is indeterminate
  *(px->psecond) = 2;

  y[0] = *(px->psecond);
  return y[0];
}

int main()
{
  int rc = union03();
  return rc;
}
