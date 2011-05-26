// Debug the destructuration of a C sequence by the new controlizer

// Sequence in a sequence: not OK in freia03.c, but OK here

// However, the structured block with the declaration of j is
// unecessarily flattened by the NEW_CONTROLIZER and C99 code is produced.

// The output of the CONTROLIZER is closer to the initial source code.

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
