/* Check that references to functions, which are constants, are not
   taken into account for memory effects

   Similar to function01, but all effects are computed
*/

int foo(int i)
{
  return i;
}

void function02()
{
  int (*f)(int) = foo;
  int j;

  if(f==foo)
    j = foo(2);
  else
    f = foo;

  return;
}
