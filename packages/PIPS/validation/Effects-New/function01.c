/* Check that references to functions, which are constants, are not
   taken into account for memory effects

   Bug found in SPEC2000/ammp.c
*/

int foo(int i)
{
  return i;
}

void function01()
{
  int (*f)(int) = foo;
  int j;

  if(f==foo)
    j = foo(2);

  return;
}
