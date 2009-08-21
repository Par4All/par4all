void flatten_code09(void)
{
  int i = 2;
  int j = 3;
  int k[3] = { 1, 2, 3 };

  i++;
  {
    int k[3] = { i, j+1, 3 };
  }
  if (1)
  {
    int k[3] = { 1, 2, 3 };
  }
}
