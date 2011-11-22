// Check that identity transformers do not lead to some information
// loss
//
// Here j = 2 before the loop, which is entered or not, and j is
// conditionnaly set to 3 within the loops. So its only two possible
// value after the loop are 2 and 3. This information is only found if
// the test k>=3 is removed

// Variation on schrammel06: the initialization is guarded by a loop
// instead of a test

main()
{
  int i, j = 2, k;

  while(i*i>1) {
    while(k>=3) {
      j = 3;
      k--;
    }
  }
  j = j;
}
