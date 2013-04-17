
int main()
{
  int a, b;
  int *p;
  
  p=&a;
  a=1;
  
  *p=0;
  
  if (a==1)
    b=1;
  else
    b=0;
  
  return b;
}
