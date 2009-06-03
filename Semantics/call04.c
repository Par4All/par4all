/* Chekc that unvisible side effects are not shown in the summary transformer */

int j = 0;

int call04(int i)
{
  i++;
  j++;
}

main()
{
  int ai = 3;
  float ax = 4.;

  ai = call04(ai++);

  ai = 0;
}
