
void inc02(int *p)
{
  *p = *p + 1;
  return ;
}

int main()
{
  int i = 0;
  inc02(&i);
  inc02(&i);
  
  return i;
}
