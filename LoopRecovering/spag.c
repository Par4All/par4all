int bazar(float c) {
  int i;
  double d = c*c;
  if (d > 2) goto big;
  else if (c < 0) {
    c = -c;
    goto neg;
  }
 big:
  i = c;
 neg:
  c += d;
 loop:
  if (i < 0) goto exit;
  i--;
 exit:
  return d;
}
