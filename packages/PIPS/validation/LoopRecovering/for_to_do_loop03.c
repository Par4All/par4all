int main() {
  int a[100];
  int i, j;

  // A do loop ! Should be parallel.
  for(j = 10; j > 0; j = -4 + j)
    a[j] = 2;

  return 0;
}
