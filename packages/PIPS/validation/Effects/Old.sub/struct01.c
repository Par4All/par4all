int main()
{
  struct one {
    int first;
    int second;
  } x, y[10], z[10];
  int i;

  x.first = 1;
  x.second = 2;

  for (i=0; i<10; i++)
    {
      y[i].first = x.first;
      y[i].second = x.second;
    }

  for (i=0; i<10; i++)
    {
      z[i] = x;
    }
  return 0;

}
