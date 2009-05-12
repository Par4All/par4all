int main() {
  int a[100];
  int i, j;

  for(i = 2; i <= 50; i = 2 + 1)
    a[i] = 2;

  // A do loop ! Should be parallel
  for(i = 2; i <= 50; i = i + 1)
    a[i] = 2;

  // A do loop ! Should be parallel. The lower bound is false but I work on it :-)
  for(j = 10; j > 0; j = -4 + j)
    a[j] = 2;

  // Should be parallel but for->do here not implemented yet...
  for(j = 60; j > 10; j = j - 3)
    a[j] = 2;

  return 0;
}
