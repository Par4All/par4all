
int inc01()
{
  int i = 0;
  int *p;
  
  p=&i;
  *p = *p + 1;
  
  return i;
}

int main()
{
  return inc01();
}