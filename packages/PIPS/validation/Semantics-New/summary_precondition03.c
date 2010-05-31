/* Make sure that summary preconditions are propagated and computed
   when functions are called from within declarations.

   In fact, the bug disappeared when the type issue between the call
   site of foo and the definition of foo was solved. No more warning
   by gcc, no more unfeasible precondition for foo.

   Also make sure that j=i++ is properly translated into a
   transformer.
 */

long long foo (char** argv) {
  long long result = 0;
  result = atoll (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  // Parsing error because i is used in its own declaration statement
  //int i = 10, j = i+1, a[i], k = foo(i);

  int i = 10;
  int j = i++, a[i];
  long long k = foo(argv);

  return j;
}
