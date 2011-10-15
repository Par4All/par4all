// Bug found in benchmark tsc.c (TSVC translated in C at UIUC, PACT 2011)

int main()
{
  int a[5][6];
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 6; j++) {
      a[i][j] = i*5 + j;
      if(i+j==4) break;
      printf("%d\n", i);
    }
  return 0;
}
