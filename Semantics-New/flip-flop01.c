int main()
{
  int s[2][10];
  int i, j;
  int cur, next;
  cur = 0;
  next = 1;
  for(i = 1; i<10; i++)
    {
    for (j = 1; j< 10; j++)
      {
	s[cur][j] = s[next][j] + j;
	s[next][j] = s[cur][j];
      }
    cur = next;
    next = 1-next;
    }
  return 0;
}
