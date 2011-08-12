/* Bug when internal addresses are also used by the programmer */

void loop_continue01()
{
  int i = 0, n = 5;
  float a[5];

  if(0) goto loop_end_1;

  for(i=0;i<n;i++) {
    if(i==0) {
      continue;
    }
      a[i] = 0.;
  }
  i = -1;
 loop_end_1:
  return;
}
