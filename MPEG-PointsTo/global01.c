/* Compute values for global variables. Bug found in
   Benchmarks/whetstones.c, but not reproduced here. */

int n1;

void zero(int *px)
{
  /* Without points-to information, this should generate a
     transformer "T(n1) {}" but function zero() does not know much
     about its compilation unit and the compilation unit does not know
     much about effects on its variable. There mimght be two ways to
     address the issue:

     1) Assume that all variable in the function and its compilation unit
     are potentially toucher and allocate old and intermediate values
     for them unconditionnally and update effects_to_transformer()

     2) Postpone the problem and handle the issue when zero
     transformer is translated into the main environment. Use zero's
     effects to fix zero's transformer.

     Solution 1 seems to fail because the potential write effect of n2
     by zero is overlooked, even though the effect on n1 can be
     derived from the compilation unit.

     Solution 2 generates a buggy transformer for "*px=o;".
 */
  *px = 0;
}

int main()
{
  int i;
  int n2 = 3;
  n1 = 0;
  zero(&i);
  i = n1;
}
