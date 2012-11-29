/* Assignment with wrong type commented out */

int main()
{
  int **c, **a, **b, *y;
  int  x = 1, z = 2, p = 1;

  //a = &x;
  b = &y;
  if (p)
    y = &z;
  else
    y = &x;
  c = &y;
  return 0;
}
