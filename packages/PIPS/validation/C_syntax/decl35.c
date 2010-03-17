/* Check that an internal function declaration does not disturb the interprocedural analyses
 *
 * Problem shown in Demo-2009/convol-unroll-C.tpips
*/

// An external declaration is properly handled here
void decl35();

main()
{
  // But it is not handled nicely here although the declaration statement is identical
  void decl35();
  int i = 1;

  decl35(i);
}

void decl35(int i)
{
  i++;
}
