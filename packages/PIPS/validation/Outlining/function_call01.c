/* An example to show that b should be declared in the function generated */

int b(i) {
  return i*2;
}

void a() {
  int i, j;
  j = 0;
 kernel:
  for(i=0;i<256;i++)
    j += b(i);
}
