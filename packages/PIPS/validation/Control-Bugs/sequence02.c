// Debug the destructuration of a C sequence by the new controlizer

void sequence02()
{
  int i = 4;

  if((i%2)==0) goto l100;
  i = i + 10;
 l100:
  i = i + 20;

  return;
}
