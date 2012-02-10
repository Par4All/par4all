int main()
{
  int i, j;
  for(i = 0; i<10; i++)
    {
      int k = i;
      {
	int l = k;
	j = j + l;
      }
    }
  return j;
}
