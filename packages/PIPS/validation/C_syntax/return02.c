/* Check that gotos are generated to represent C return statements,
 *  except when there is a single return located at the end of the
 * function
 */

void return02()
{
  int i = 1;
  i++;
  return;
}
