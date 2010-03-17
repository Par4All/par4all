void decl41(int size)
{
  int i;
  int x;
  for(i=1 ; i<size ; i++)
    x = i;

  // if the switch is removed, the loop index declarations reappears
  switch(0) {
  case 0:
    x = 1;
      ;
  }
  x = 0;
}
