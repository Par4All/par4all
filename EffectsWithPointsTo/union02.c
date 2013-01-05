/* Pointer version of union01 */

int union02()
{
  union one {
    int first;
    int second;
  } x, *px;
  int y[10];

  px = &x;
  px->first = 1;
  px->second = 2;

  y[0] = 1;
}
