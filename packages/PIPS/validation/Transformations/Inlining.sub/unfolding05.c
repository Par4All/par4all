/* Bug in unfolding when the inlined function is incorrect because it
   does not contain anything.

   This happens in PIPS when code is synthesized by PIPS. This might
   be fixed, but in between the behavior of the unfolder should be
   improved when this happens:

   user warning in inline_expression_call: expanded sequence_statements
   seems empty to me

   is more than a warning because the generated code is not parsable.

*/

int unfolding05(int i)
{
  //return i+1;
}

int main()
{
  int j;

  j = unfolding05(4);
}
