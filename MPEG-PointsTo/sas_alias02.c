// SAS 2010, Alias & al., Fig. 2.a, p. 125

// To prove termination

float alea(void)
{
  return 1.;
}

void sas_alias02(int m)
{
  int x=m, y=m;
  while(x>=2) {
    if(1) {
      x--, y+=x;
      while(y>=x && alea()>=0.)
	y--;
      x--, y-=x;
    }
  }
}

