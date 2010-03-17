/* Check that static functions are recorded with names that can be
   used to retrieve them later: build the call graph */

static void foo(int i)
{
  i++;
}

int main()
{
  foo(3);
}
