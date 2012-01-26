static int count;

void foo()
{
  count++;
}

int main(void)
{
  foo();
  foo(1);
  foo(1, 2);
  return count;
}
