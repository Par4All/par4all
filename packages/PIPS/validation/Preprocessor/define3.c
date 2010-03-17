/* Prettyprint issue with unary plus next to binary plus: do not make
   it an increment operator. */

main()
{
  int tmp;

  tmp = -1 + +1;
  return tmp;
}
