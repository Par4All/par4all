/* Chekc that unvisible side effects are not shown in the summary transformer */

int j = 0;

void call03(int i, float x)
{
  i++;
  j++;
  x++;
}

main()
{
  int ai = 3;
  float ax = 4.;

  call03(ai, ax);

  ai = 0;
}
