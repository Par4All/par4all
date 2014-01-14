
int main()
{
  int a, b;
  int *p;
  
  p=&a;
  a=42;
  b=-1;
  
  *p=1;
  
  if (a==1)
    b=1;
  else
    b=0;
  
  return b;
}
