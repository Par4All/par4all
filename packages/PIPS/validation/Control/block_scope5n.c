/* Jump into a block from a goto before the block
 *
 * The internal declaration "int x=7;" is badly placed and the
 * generated code does not compile.
 *
 * It is not clear if it is a controlizer issue or a prettyprinter
 * issue. It looks like a prettyprinter issue because the "x--;" is
 * moved up because of the "goto lab1;". However, the controlizer is
 * supposed to move the declaration upwards...
 */

void block_scope5n()
{
  int i;
  {
  int x = 6;
  goto lab1;
 lab2:
  x = 2;
  {
    int x = 7;
  lab1:
    x--;
  }
  x++;
  goto lab2;
  }
}
