// single array element assignments
int main()
{
  int *a[2];
  int b = 0;
  int c = 1;
  a[0] = &b;
  a[1] = &b;
  a[1] = &c;
  return(0);
}
