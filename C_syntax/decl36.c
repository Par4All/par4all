/* Check that a function can be redeclared with the same arguments,
 * when a file is included more than once for instance.
 *
 * Problem shown in C_syntax/tpips.tpips
*/

void decl36(int i);

void decl36(int i);

main()
{
  int i = 1;

  decl36(i);
}

void decl36(int i)
{
  i++;
}
