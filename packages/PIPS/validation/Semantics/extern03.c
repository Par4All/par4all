/* Check that successive external floating point dependent initializationa are properly taken
   into account. */

float delta = 1.;
float delta2 = delta+2.;

main()
{
  float i = 0.;

  i = i + delta;
  i = i + delta;
  i = i + delta2;
}
