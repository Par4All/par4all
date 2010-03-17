/* Chekc that values returned by functions are exploited */

int call06(int i)
{
  i++;
  return i;
}

main()
{
  int ai;

  ai = call06(2);
}
