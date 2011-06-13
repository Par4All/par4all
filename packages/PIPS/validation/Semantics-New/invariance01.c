/* To investigate the surprise found in induction_substitution03.c
 *
 * b==a#init is now lost?
 *
 * Because somebbody added a filtering of initial values in
 * statement_to_postcondition(), probably to avoid a bug in regions...
 */

int invariance01(int a)
{
  int b;

  a = a + 1;
  a = a - 1;
  b = a;
  a = 2;
  return b;
}
