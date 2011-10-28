// PIPS prettyprinter: copy of if04.c, with ambiguous else branch

int main()
{
  int i, c= 0;

  if(c>1)
    if(c>2) {
      if(c>3)
	i =1;
    }
    else
      i= 2;

  return i;
}
