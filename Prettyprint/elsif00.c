int elsif00(int i)
{
  // direct
  if (i==1)
    i++;
  else if (i==2)
    i--;
  else
    i-=3;
  // blocks
  if (i==4) {
    i++;
  } else if (i==5) {
    i--;
  } else {
    i-=6;
  }
  // sub blocks
  if (i==2) {
    i++;
  } else {
    if (i==5) 
      i--;
    else
      i-=3;
  }
  // block sub blocks
  if (i==2) {
    i++;
  } else {
    if (i==5) {
      i--;
    } else {
      i-=3;
    }
  }
}
