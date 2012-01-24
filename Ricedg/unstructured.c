// This test case used to make pips ricedg phase to fail because of the
// unstructured. It is in the validation to avoid a regression

// In fact, the failure was due to expression instructions created by
// the controlizer when the for loop is controlized because of the
// conditional break. This test case is useful to check that
// expression instruction are properly processed.

int StopCondition(void) {
  return 1;
}

int main (int argc, char** argv) {
  int i;
  for ( i = 1; i <= 100; i++) {
    if (StopCondition () == 1) {
      break;
    }
  }
  return 0;
}
