// PIPS prettyprinter does not regenerate the superfluous braces
// for the true branch of the test, which leads to a warning by gcc,
// even though the input code is OK with gcc. This can be avoided with
// Property PRETTPRINT_GCC_C_BRACES.

int main()
{
  int i, c= 0;

  if(c>1)
    if(c>2) {
      if(c>3)
	i =1;
    }
    else
      i= 2;

  return i;
}
