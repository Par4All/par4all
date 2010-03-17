int ex2(int argc, char* args)
{
  int U[101][101];
  int V[101][101];

  int i, j;

  for(i = 0; i < 100; i++)
    {
      for(j = 0; j < 100; j++)
	{
	  U[i+1][j] = V[i][j] + U[i][j];

	  V[i][j+1] = U[i+1][j];
	}
    }
  return 0;
}
