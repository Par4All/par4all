/* handling of cast */

int main()
{
  int d1 = 4, d2 = 4;
  int y[d1][d2];
  int * p = (int *) y;

  return *p;
}
