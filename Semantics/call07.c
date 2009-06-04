/* Chekc that values returned by functions are exploited */

int call07(int i)
{
  return i++;
}

main()
{
  int ai;

  ai = call07(2);
}
