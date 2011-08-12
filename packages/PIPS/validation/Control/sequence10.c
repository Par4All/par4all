// Bug inherited from FREIA after unfolding. See Ticket 555.

// Some code seems to be lost by the controlizer: yshift--;

// Note: the parsed printed file is badly printed too; it cannot be
// compiled because label break_1 is repeated...


int foo(n)
{
  return n+14;
}

void sequence10(int argc, char *argv[])
{
  int i, j;
  double a[5][5];

  for(i=0;i<10;i++) {
    if(i==3) break;
    for(j=0;j<10;j++)
      a[i][j]=foo(j);
  }
  return;
}

int main(int argc, char * argv[])
{
  sequence10(argc, argv);
}
