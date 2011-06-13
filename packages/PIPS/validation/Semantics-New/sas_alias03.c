// SAS 2010, Alias & al., Fig. 2.b, p. 125

// To prove termination

float alea(void)
{
  return 1.;
}

void sas_alias03(int m)
{
  int x=m, y=m;
  while(x>=2) {
    if(1) {
      x--, y+=x;
      while(y>=x+1 && alea()>=0.) {
	if(1) {
	  y--;
	  while(y>=x+3 && alea()>=0.)
	    x++, y-=2;
	  y--;
	}
      }
      x--, y-=x;
    }
  }
}

