// Bug inherited from FREIA after unfolding. See Ticket 555.

// Some code seems to be lost by the controlizer

// To start with, the initialization of y0 is lost, then the following block...

void sequence07(int argc, char *argv[])
{
  int xshift, yshift;

  if (0) goto _break_1;

  {
    int *x0 = &xshift, *y0 = &yshift;
    {
      int i;
      for(i = 0; i <= 9; i += 1)
	// commenting next lines makes unfolding work...
	*x0 = 18;
      goto l99998;
    l99998: ;
    }
  }

 _break_1:   ;
  return;
}

int main(int argc, char * argv[])
{
  sequence07(argc, argv);
}
