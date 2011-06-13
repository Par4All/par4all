// To check the conversion of if in to while

void whileif01()
{
  int i = 0;

  while(i<=10) {
    if(i>=0)
      i++;
    else
      i--;
  }

  i = -1;
  while(i<=10) {
    if(i>=0)
      i++;
    else
      i--;
  }
  return;
}
