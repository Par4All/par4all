int main()
{
  int i = 1;
  for (int a[5] = {1, 42, 67, 90, 7}; i < 5; i++) {
    a[i] = i;
    printf("%d\n", i);
  }
  return 0;
}
