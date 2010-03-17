/* Make sure that summary transformers are consistent */

int j = 0;

int call12(void)
{
  j++;

  return j;
}

main()
{
  int ai = 3;

  ai = call12();

  ai = call12();

  ai = 0;
}
