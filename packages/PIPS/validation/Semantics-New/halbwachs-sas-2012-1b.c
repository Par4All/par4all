// Halbwachs & Henry, SAS 2012, p. 199

void foo()
{
  int i = 0, j;
  while(i<100) {
    j = 0;
    while(j<100)
      j++;
    i++;
  }
  return;
}
