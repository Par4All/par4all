/* Check simplifications of boolean expressions
 *
 * The precondition after the first test must be correct (2<=i<=3), but the test
 * condition is not simplified by suppress_dead_code.
 *
 * Also, the test structure is not simplified by simd_atomizer.
 *
 * But the second test is perfectly simplified.
 *
 * Needed for Vivian Maisonneuve
 */

int main() {
  int a = 1;
  int b;
  int i;

  if(a>0 && b>0)
    i = 2;
  else
    i = 3;

  if(a>0 || b>0)
    i = 2;
  else
    i = 3;

  return i;
}
