void decl38 (int size)
{
  int i;
  int j;
  int k;
  float x;

  for(i=1 ; i<size ; i++)
    ;

  // if the switch is removed, the loop index declarations reappears
  switch(0) {
  case 0:
    for(j=1 ; j<size ; j++)
      ;
  }

  for (k=1; k<size; k++)
    ;
  x = 0;
}
