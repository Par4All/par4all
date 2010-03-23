/* Check what happens with a name conflict solved with a scope. The
   internal a should be a copy of the parameter and the function
   should return a+1, with a its formal parameter. It does not
   work. The problem may occurs within the parser because a is
   declared before the initialization expression is parsed.  */

int initialization06(int af)
{
  int a = af;
  int i;

  i = a+1;
  a = 2;
  return i;
}
