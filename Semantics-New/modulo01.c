/* Bug: side effects ignored in modulo based expressions. see
   modulo_to_transformer(). */

main()
{
  int i;
  int j;

  j = (i++)%2;
}
