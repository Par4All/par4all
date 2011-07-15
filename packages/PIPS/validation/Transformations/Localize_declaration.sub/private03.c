int main()
{
  int i,j;
  int tmp;
  int a[10][10];

  // tmp must be declared as private only in the second loop
  for(i=0; i<10; i++)
    {
      for (j=0; j<10; j++)
	{
	  tmp = a[i][j];
	  a[i][j] = tmp+j;
	}
    }
  return 0;

}
