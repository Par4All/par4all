// pointer arithmetic
int main()
{
  int a[10];
  int *p;
  int b = 0, c = 1;
  if(b==c)
    p = &a[0];
  else
    p = &a[1];
  p = p+1;
  p = p+3;
  p = p-1;
  p = p-2;
  return(0);
}
