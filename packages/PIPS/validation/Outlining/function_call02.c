/* An example to show that b should be declared in the function generated */

int b(i) {
  return i*2;
}

void a() {
  int i, j, k;
  j = 0;
  for(k=0;k<256;k++)
  kernel:
    for(i=0;i<256;i++)
      j += b(i);
}
