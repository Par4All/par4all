int main()
{
  int i, j;
  for(i = 0; i<10; i++)
    {
      int k = i;
      {
	int l = 2;
	{
	  int k = 3;
	  l = l + k;
	}
	k = k + l;
      }
      j = j + k;
      {
	int l = 4;
	j = j + l;
      }
    }
  return j;
}

