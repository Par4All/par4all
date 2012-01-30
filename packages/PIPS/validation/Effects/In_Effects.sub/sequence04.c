int main()
{
  int a[2], b, c;

  a[0] = 0;
  a[1] = 1;
  if(1)
    {
      a[1] = 2;
      b = a[1];
    }

  a[0] = 2;
  a[1] = 3;
  if(1)
    {
      b = a[0] + a[1];
    }
  return b;
}
