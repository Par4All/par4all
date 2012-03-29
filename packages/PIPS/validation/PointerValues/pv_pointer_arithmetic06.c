// pointer arithmetic on 2D array

int main()
{
  int a[1][2], b, *p;
  a[0][0] = 0;
  a[0][1] = 1;
  if (a[0] > a[1])
    {
      p = a;
      b = 0;
    }
  else
    {
      p = a + 1;
      b = 1;
    }
  b = p[0];
  return b;
}
