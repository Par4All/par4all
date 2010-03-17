int main() {
  int a[100];
  int i, j;

  // A do loop ! Should be parallel
  for(i = 2; i <= 50; i = i + 1)
    a[i] = 2;

  /* The following code should not prevent the previous loop to be
     parallel: */
  // Should be parallel but for->do here not implemented yet...
  for(j = 60; j > 10; j = j - 3)
    a[j] = 2;

  for(i = 2; i <= 50; i = 2 * i)
    a[i] = 2;

  return 0;
}
