/* Check that references to functions, which are constants, are not
   taken into account for memory effects

   Similar to function01 and 02, but all effects are computed and f is used
*/

int foo(int i)
{
  return i;
}

void function03()
{
  int (*f)(int) = foo;
  int j;

  if(f==foo)
    j = foo(2);
  else
    f = foo;

  // Two possible syntaxes (?) to use pointer f
  j = f(2);
  j = (*f)(2);

  return;
}
