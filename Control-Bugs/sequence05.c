// Debug the destructuration of a C sequence by the new controlizer

// sequence in a sequence: not OK in freia03.c, but OK here

void sequence05()
{
  int i = 4;

  if((i%2)==0) goto l100;
  i = i + 10;
  {
  l100:

    i = i + 20;
    {
      int j = 3;
      i += j;
      goto here;
    here: ;
    }
    i+=30;
  }
  return;
}
