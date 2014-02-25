int maxim(int a, int b)
{
  if(a<b) return b;
  return a;
}

int constant()
{
  return 42;
}

int main(void)
{
  return  maxim(constant(), 0);
}
