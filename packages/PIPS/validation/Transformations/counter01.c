int compute(int n) {
  int i = 1;
  while (i<n) {
    i<<=1;
    if (rand()) i++;
  }
  return i;
}
