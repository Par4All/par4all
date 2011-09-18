// Simplify control: the conditional operator is not handled because
// the precondition has to be asserted within an
// expression. load_statement_precondition cannot help.

#include <assert.h>

void sequence04()
{
  int i = 4;

  /* This assert cannot be cleaned up using preconditions only */
  assert(4>3);

  return;
}
