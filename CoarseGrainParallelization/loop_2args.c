#define INIT_X  5
#define INIT_Y  5
#define SIZE    100
#define MACSIZE 16

int main () {
  int y = 0;
  int x = 0;
  int yS = 0;
  int xS = 0;
  int a [SIZE][SIZE];

  // the two loop nests do the same things but only the second one is
  // paralellized
  for(yS=MACSIZE,y=INIT_Y; y<(INIT_Y+MACSIZE), yS<(2*MACSIZE); y++,yS++)
    {
      for(xS=MACSIZE,x=INIT_X; x<(INIT_X+MACSIZE), xS<(2*MACSIZE); x++,xS++)
	{
	  unsigned char pxSource;
	  pxSource = a[yS][xS];
	}
    }

  for(y=INIT_Y; y<(INIT_Y+MACSIZE); y++)
    {
      yS=MACSIZE- (y - INIT_Y);
      for(x=INIT_X; x<(INIT_X+MACSIZE); x++)
	{
	  xS=MACSIZE - (x - INIT_X);
	  unsigned char pxSource;
	  pxSource = a[yS][xS];
	}
    }
}
