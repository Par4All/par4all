/* Check that successive initializations are properly taken into
 * account.
 *
 * Special abnormal case: i is a constant by definition (bug spotted by
 * Serge Guelton: because i is never written, its initial and only
 * assignment cannot be processed like an assignment). Definitely not
 * a priority bug...
 *
 * In c_data_to_prec_for_variables(), any_expression_to_transformer()
 * should be used instead of
 * safe_assigned_expression_to_transformer(). Something like:
 *
 * tf = any_expression_to_transformer(tmpv, sub_exp, pre, FALSE);
 *
 */
#include <stdio.h>
void static04()
{
  static int i = sizeof(i);

  printf("%d\n", i);
}

main()
{
  static04();
  static04();
  static04();
}
