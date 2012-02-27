/* ROW used twice as typdef, for a dependent type! */

/* ROW is no longer a dependent type and PIPS preprocessor still fails... */

void local_typedef12a(int n) {
  /* Create alternate buffer: m rows of n. */
  typedef int ROW[10];
  ROW x;
  x[1] = 0;
}


void local_typedef12b(int n) {
  typedef int ROW[10];
  ROW x;
  x[1] = 0;
}
