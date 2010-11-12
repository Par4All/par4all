/* Check that gotos are generated to represent C return statements,
 *  except when there is a single return located at the end of the
 * function.
 *
 * Complement of return02.c to make sure the goto is preserved in this case
 */

void return05()
{
  int i = 1;
  i++;
  return;
  i++;
}
