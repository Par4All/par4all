/* Chekc that visible side effects are taken into account in the
   caller's transformer when the returned value is unanalyzable and
   ignored */

int j = 0;

double call10(int i)
{
  double x = 3.;
  i += 10;
  j++;

  return x;
}

main()
{
  int ai = 3;

  call10(++ai);

  ai = 0;
}
