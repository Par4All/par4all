/* Check the impact of anymodule:anywhere on the semantics analysis
 *
 * Simplified version of anywhere01 to check the analysis of a comma expression
 *
 * The analysis fails in safe_expression_to_transformer() because the
 * expression effects are not computed correctly in presence of
 * pointer dereferencements.
 */

void anywhere03()
{
  int m = 1, n = 17;
  int *p = &m;

  /* Impact of comma expression */
  *p = 19, n = 2;

  return;
}
