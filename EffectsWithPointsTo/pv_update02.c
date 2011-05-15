// update operators
int main()
{
  int a[10];
  int *p;
  int b = 0, c = 1;
  if(b==c)
    p = &a[0];
  else
    p = &a[1];
  p++;
  ++p;
  p--;
  --p;
  return(0);
}
