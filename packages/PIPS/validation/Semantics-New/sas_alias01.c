// SAS 2010, Alias & al., Fig. 1, p. 121

// To prove termination

float alea(void)
{
  return 1.;
}

void sas_alias01(int m)
{
  int x=m, y=0;
  while(x>=0 && y>=0) {
    if(alea()>=0.) {
      while(y<=m && alea()>=0.)
	y++;
      x--;
    }
    y--;
  }
}

