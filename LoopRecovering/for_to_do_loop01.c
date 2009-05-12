int main() {
  int a[100];
  int i, j;

  // A do loop ! Should be parallel
  for(i = 2; i < 100; i += 1)
     a[i] = 2;

  // Should not be a do loop because j is not the expected index !
  for(i = 2; i <= 50; j += 1)
     a[i] = 2;

  return 0;
}
