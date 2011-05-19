// Example 9.64 p. 699 in Aho-Lam-Sethi-Ullman

// FI: I do not see the direct relationship woth induction variable
// detection. The optimization is to write b = a + j - 1 and to
// eliminate the updates of a, but the code is a bit contrived and not
// exciting... The forward substitution and the invariant code motion
// may not be sufficient to optimize such a piece of code...

// The invariance of a in the loop is available from the loop
// transformer, but not from the precondition. It could be available if
// "a" were a formal argument, but this is not found in the loop
// postcondition, which is surprising... The precondition a==a#init is
// implicit and is not taken into account when a transformer is
// applied to a precondition. The implicit equation, a==a#init, of any
// transformer or precondition where a is not argument seems to be forgotten.

// Since b is not used in such a short text book example, the whole loop can be
// replaced by b = a + 8. This is found in the precondition for
// "return" but the uselessness of the loop cannot be proven without
// its transformer T(a) {a==a#init}.

// The summary transformer is nevertheless correct, which indicates
// that the invariance of a is found.

#include <stdio.h>

int induction03(int a)
{
  int /*a,*/ b, i, j;

  scanf("%d", &a);

  for(j=1; j < 10; j++) {
    a = a - 1;
    b = j + a;
    a = a + 1;
  }
  return b;
}
