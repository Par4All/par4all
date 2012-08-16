/* Excerpt from array10, function bar(), slightly modified
 *
 * Bug: the IN context does not seem always updated... but it should
 * not be updated here because the parameter p is passed by value. p
 * is then updated in function array13, but this has no impact on the
 * calling function. The initial value of p is lost. No need for an IN
 * context.
 */

int array13(int *p)
{
  int b[100];
  p = &b[0];

  return 0;
}
