/* Make sure that summary preconditions are propagated and computed when functions
   are called from within declarations */

long long foo (char** argv) {
  long long result = 0;
  result = atoll (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  // Parsing error because i is used in its own declaration statement
  //int i = 10, j = i+1, a[i], k = foo(i);

  int i = 10;
  int j = i++, a[i], k = foo(i);

  return j;
}
