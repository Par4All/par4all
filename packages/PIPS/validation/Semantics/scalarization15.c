/* Expected result:
   x[i] and t[i] can be scalarized

   NOTE: This is almost identical to scalarization14: no copy-out on
   x[i] and t[i], as only y is copied out.

   BUG on return statement :

   user warning in transformers_intra_fast: Property SEMANTICS_FLOW_SENSITIVE is ignored
   user warning in add_or_kill_equivalenced_variables: storage return
   user warning in NormalizeExpression: expression is already normalized
   user warning in NormalizeExpression: expression is already normalized
   user warning in NormalizeExpression: expression is already normalized
   pips error in value_to_variable: *SEMANTICS*:t#0 is not a non-local value

 */


#include <stdio.h>
#define SIZE 100

int scalarization15(int n)
{
  int x[SIZE], y[SIZE][SIZE], t[SIZE];
  int i, j;

  for (i=0 ; i < SIZE ; i++) {
    x[i] = i;
    for (j=0 ; j < SIZE ; j++) {
      t[i] = x[i];
      y[i][j] = x[i] + j;
    }
  }
  return y[n][n] + y[0][0] + y[0][n] + y[n][0];
}

int main(int argc, char **argv)
{
  printf("%d\n", scalarization15(5));
}
