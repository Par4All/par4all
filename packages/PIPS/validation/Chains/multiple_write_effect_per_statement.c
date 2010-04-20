

void multiple() {
  int i,j;

  i = 0, j= 0; // Chains used to never kill these kind of statements

  i = 1; // This one should kill previous assignment to i

  j = i; // There should be only one incoming true dependence
}
