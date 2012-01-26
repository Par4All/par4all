/* Requested by Corinne Ancourt to check handling of global effects */

extern int i;

int j;

static int k;

void inter01()
{
  static int l;
  int m;
  int n;

  for(n=0; n<10; n++) {
    i = 1;
    j = 2;
    k = 3;
    l = 4;
    m = 5;
  }
}

main()
{
  inter01();
}
