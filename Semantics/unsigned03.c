int foo(int i, unsigned int u)
{
  return i + u;
}

int main(void)
{
  int j;

  j = foo(-5, 12);

  return j;
}
