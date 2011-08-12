/* Bug when internal addresses are also used by the programmer */

void loop_break01()
{
  int i = 0, n = 5;
  float a[5];

  if(0) goto break_1;

  for(i=0;i<n;i++) {
    if(i==0) {
      a[i] = 0.;
      break;
    }
  }
  i = -1;
 break_1:
  return;
}
