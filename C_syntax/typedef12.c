

void local_typedef1(int n) {
  // Create alternate buffer: m rows of n.
  typedef int ROW[n];
  ROW x;
  x[1] = 0;
}


void local_typedef2(int n) {
  typedef int ROW[n];
  ROW x;
  x[1] = 0;
}
