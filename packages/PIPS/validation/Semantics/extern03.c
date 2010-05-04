/* Check that successive external floating point dependent
   initializations are properly taken into account.

   Transformers are computed "in context". Make sure that the
   precondition is used in the declarations.
 */

float delta = 1.;
float delta2 = delta+2.;

main()
{
  float i = 0.;
  int k = 1;

  i = i + delta;
  i = i + delta;
  i = i + delta2;
}
