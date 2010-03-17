/* Chekc that values returned by functions are exploited */

int call05(int i)
{
  return i;
}

main()
{
  int ai;

  ai = call05(2);
}
