int main()
{
  int a[5][6];
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 6; j++) {
      a[i][j] = i*5 + j;
      printf("%d\n", i);
    }
  return 0;
}
