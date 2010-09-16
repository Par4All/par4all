// Check for lack of predicates in schrammel04.c, NSAD 2010

main()
`{
  int i, j = 2, k;

  while(i*i>1) {
    while(k*k>1)
      j = 1;
    j = j;
  }
  j = j;
}
