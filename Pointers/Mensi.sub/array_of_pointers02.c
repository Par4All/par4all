int main()
{
  int *a[10];
  int b = 0, c = 1;
  a[0] = &b;
  a[1] = &b;
  a[1] = &c;
  return 0;
}
