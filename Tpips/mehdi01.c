/* Ticket 280:
 *
 * Default activation for transformers:
 *
 * tpips mehdi01.tpips: intraprocedural
 *
 * tpips <mehdi01.tpips: interprocedural
 *
 */

void mehdi01()
{
  int i = 1;
  int j;
  j = i;
  return;
}
