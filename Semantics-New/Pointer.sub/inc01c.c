
int inc01c()
{
  int i = 0;
  int *p;
  
  p=&i;
  (*p)+=1;
  
  return i;
}

int main()
{
  return inc01c();
}