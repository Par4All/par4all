int main()
{
  int i;
  int a[10];

  for(i=0; i<10; i++)
    {
      a[i] = i;
    }
  for(i=0; i<10; i++)
    {
      a[(int)i] = i;
    }

  return 0;
}
