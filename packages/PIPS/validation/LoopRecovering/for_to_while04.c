int main() {
  int a[100];
  int i, j;

  /* Test with incomplete for: */
  i = 2;
  for(; i <= 50; i++)
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  for(i = 2; i <= 50;) {
    i++;
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;
  }

  for(i = 2; ; i++)
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  /* Unreachable from here... */
  for(;;)
    for(j = 2; j < 100; j *= 2)
      a[j] = 2;

  return 0;
}
