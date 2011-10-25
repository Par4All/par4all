int main()
{
  int i,j,t;
  int u[10][10], res;
  for (t = 1; t<100; t++)
    {
      for (i = 0; i <10; i++)
	for (j = 0; j < 10; j++)
	  u[i][j] = 0.0;

      for (i = 0; i <10; i++)
	for (j = 0; j < 10; j++)
	  res = res + u[i][j];
    }
  return res;
}
