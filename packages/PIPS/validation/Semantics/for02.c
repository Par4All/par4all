int main()
{
  int i,j,k;
  int a[500];

  for (i=0, j=1; i<=499; i++, j++) {
    // Cumulated effects should be a[i] here and not a[*]
    a[i] = i;
  }
  // We have j==501 in the preconditions here... since we use
  // the proper activate and properties
  k = 2;
  return k;
}
