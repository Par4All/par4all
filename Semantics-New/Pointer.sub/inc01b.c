
int inc01b()
{
  int i = 0;
  int *p;
  
  p=&i;
  (*p)++;
  
  return i;
}

int main()
{
  return inc01b();
}