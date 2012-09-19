/* Example developped to check Amira's small language */

struct s { int ** pp1; int **pp2;};

void foo(struct s *ps)
{
  int i, j;
  *((*ps).pp1) = &i;
  *((*ps).pp2) = &j;
  **((*ps).pp1) = 1;
  **((*ps).pp2) = 2;
  **((*ps).pp1) = 3;
  return; // to get the final points-to graph
}
