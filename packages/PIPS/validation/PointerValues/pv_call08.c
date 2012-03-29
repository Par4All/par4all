// external function call with no pointer involved, and non-pointer return value


int foo(int a)
{
  return a;
}

int main()
{
  int a;
  a = foo(2);
  return 0;
}
