/* Chekc that visible side effects are taken into account in the caller's transformer */

int j = 0;

int call08(int i)
{
  i += 10;
  j++;

  return i;
}

main()
{
  int ai = 3;

  ai = call08(++ai);

  ai = 0;
}
