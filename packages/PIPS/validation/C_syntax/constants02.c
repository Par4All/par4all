/* To check 32 bit unsigned constant...
   and what will be wrong in the preconditions */
main() {
  unsigned int i = 4294967295U;
  unsigned int j;
  int k;
  int t;
  int u;

  j = i;
  k = j;
  t = (i - j) + k;
  i = u;
  j = u;
  k = u;
}
