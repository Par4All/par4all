int main()
{
  int a, b, c;

  a= 0;
  b = 1;
  {
    int d = 1;
    c = a + d;
    a = 2;
  }
  return b;
}
