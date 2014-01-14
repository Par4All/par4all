/* Union without pointer */

int union01()
{
  union one {
    int first;
    int second;
  } x;
  int y[10];

  x.first = 1;
  x.second = 2;

  y[0] = 1;
}
