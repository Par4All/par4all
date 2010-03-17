/* To track down an infinite restructuring bug in control/unspaghettify.c */
int
desugar_loop(int i) {
  int j = 0;

  while (j < i) {
    if (j > 100)
      break;
  }
  return i;
}
