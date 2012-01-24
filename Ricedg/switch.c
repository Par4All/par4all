// This test case used to make pips ricedg phase to fail because of
// the expression case was not handled.
// It is in the validation to avoid a regression

int main (int argc, char** argv) {
  int i, j, k;
  int a[10];
  i = 0;
  j = 0;
  k = 0;
  switch (argc) {
  case 2:
    for (i = 2; i < 10; i++) {
      a[i] = i;
    }
    break;
  case 6:
     for (j = 6; j < 10; j++) {
      a[j] = j;
    }
   break;
  default:
    for (k; k < 10; k++) {
      a[k] = k;
    }
    break;
  }
  return 0;
}
