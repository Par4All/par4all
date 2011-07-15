int main()
{
  int i,j;
  int tmp;
  int a[10][10];

  // tmp must be declared as private in the two loops
  for(i=0; i<10; i++)
    {
      for (j=0; j<10; j++)
	{
	  tmp = a[i][j];
	  a[i][j] = tmp+j;
	}
      tmp = i*2;
      a[i][i] = a[i][i] + tmp;
    }
  return 0;
}
