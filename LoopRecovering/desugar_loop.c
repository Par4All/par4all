/* Try to see what the desugaring + loop recovering can do. */
int
desugar_loop(int i) {
  int j = 0;

  while (j < i) {
    j = j + 1;
    // This test should be structured at the end:
    if (j > 100)
      continue;
    j += i;
  }
  while (j < i) {
    j = j + 1;
    if (j > 100)
      break;
    j += i;
  }

  while (j < i) {
    j = j + 1;
    if (j > 100)
      break;
    if (j > 50)
      continue;
    j += i;
  }
  return i;
}
