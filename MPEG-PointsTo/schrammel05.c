// Check for lack of predicates in schrammel04.c, NSAD 2010

main()
{
  int i, j = 2, k;
  float x;

  while(i*i>1) {
    while(k*k>1)
      j = 1;
    // Oops, this generate an indentity transformer that is not identified
    //    j = j;
    x = 0.;
  }
  j = j;
}
