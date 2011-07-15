int main()
{
  int i,j;
  int tmp;
  int a[10][10];

  // tmp must be declared as private for this loop
  // even if it belongs to the loop locals of the inner loop which is not parallel
  for(i=0; i<10; i++)
    {
      a[i][0] = i;
      for (j=1; j<10; j++)
	{
	  tmp = a[i][j-1];
	  a[i][j] = tmp+j;
	}
    }
  return 0;
}
