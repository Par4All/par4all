int main()
{
  int a, b, c, *p;
  a = 0;
  b = 1;
  if (a>b)
    p = &a;
  else
    p = &b;
  c = *p;
  return b;
}
