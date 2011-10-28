// PIPS prettyprinter does not regenerate the superfluous braces
// for the true branch of the test, which leads to a warning by gcc,
// even though the input code is OK with gcc. Use property for
// compatibility with gcc, PRETTYPRINT_FOR_GCC.

int main()
{
  int i, c= 0;

  if(c>1)
    if(c>2) {
      if(c>3)
	i =1;
      else
	i= 2;
    }

  if(c>1)
    if(c>2) {
      if(c>3)
	i =1;
    }
    else
      i= 2;

  if(c>1) {
    if(c>2)
      i =1;
    else
      i= 2;
  }

  if(c>1)
    if(c>2)
      i =1;

  if(c>2)
    i =1;

  return i;
}
