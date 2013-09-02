/* Variant of use_def_elim10, to check the interprocedural advantage
 * of dead code elimination with OUT regions...
 *
 * This does not work here because OUT regions do not cope with
 * returned value. They cannot tell if the value returned by
 * use_def_elim11 is used or not.
 */

int use_def_elim11(int i)
{
  int j;
  int k;

  j = i + 1;
  i = 2;
  k = 3;
  return j;
  //return i;
  //return k;
} 

int main()
{
  int u = use_def_elim11(2);
  return 0;
}
