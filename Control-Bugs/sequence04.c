// Debug the destructuration of a C sequence by the new controlizer

// Same as sequence03, but with a declaration in the sub-block

void sequence04()
{
  int i = 4;

  if((i%2)==0) goto l100;
  i = i + 10;
  {
    int i;
  l100:

    i = i + 20;
  }
  return;
}
