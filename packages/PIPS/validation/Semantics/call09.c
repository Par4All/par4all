/* Chekc that visible side effects are taken into account in the
   caller's transformer when the returned value is analyzable and
   ignored */

int j = 0;

int call09(int i)
{
  i += 10;
  j++;

  return i;
}

main()
{
  int ai = 3;

  call09(++ai);

  ai = 0;
}
