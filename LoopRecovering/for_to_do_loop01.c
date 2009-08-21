int main() {
  int a[100];
  int i, j;

  // A do loop ! Should be parallel
  for(i = 2; i < 100; i += 1)
     a[i] = 2;

  /* The following code should not prevent the previous loop to be
     parallel: */
  // Should not be a do loop because j is not the expected index !
  for(i = 2; i <= 50; j += 1)
     a[i] = 2;

  return 0;
}
