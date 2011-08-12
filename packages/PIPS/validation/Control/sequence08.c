// Bug inherited from FREIA after unfolding. See Ticket 555.

// Some code seems to be lost by the controlizer: yshift--;

// Note: the parsed printed file is badly printed too; it cannot be
// compiled because label break_1 is repeated...


void sequence08(int argc, char *argv[])
{
  int xshift, yshift;

  if (0) goto _break_1;

  {
    int *x0 = &xshift;
    {
      int i;
      for(i = 0; i <= 9; i += 1)
	// commenting next lines makes unfolding work...
	*x0 = 18;
      goto l99998;
    l99998: ;
    }
  }
  yshift--;
 _break_1:   ;
  xshift++;
  return;
}

int main(int argc, char * argv[])
{
  sequence08(argc, argv);
}
