void flatten_code08(void)
{
  int i = 2;
  int j = 3;
  int k[] = { 1, 2, 3 };

  i++;
  {
    int k[] = { i, j+1, 3 };
  }
  if (1)
  {
    int k[] = { 1, 2, 3 };
  }
}
