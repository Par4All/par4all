// external function call with no pointer involved, and non-pointer return value
// but with pointers in caller.


int foo(int a)
{
  return a;
}

int main()
{
  int a, *p;
  p = &a;
  a = foo(2);
  return 0;
}
