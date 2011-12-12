// Goal make sure that preconditions and regions stay simple when a
// simple set of constraints exists.

// This example was designed folowing the linked_region cases.

// The chopping of one corner element could occur on any of the four
// corners. Here, corner (99,99) is not removed as in chopped_square14,
// but a convex hull is preserved.

// The worry: integer divisions are not handled as well as they
// should/could by regions

// Normalization issue? Or use expression_to_transformer(phi, N/2,
// pre) to retrieve constraints on phi?

// Also, not even a gentle sc_bounded_normalization() is applied to
// propagate constants in the region associated to A[k]...

int main()
{
  int k;
  int N = 100;
  double A[100];

  if(1) {
    k = N/2;
    A[N/2] = 0.;
    A[k] = 1.;
  }

  return 0;
}
