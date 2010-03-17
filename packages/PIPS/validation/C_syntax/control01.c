// Break in enclosed control structures

int control01(int *in, int *val)
{
 int c;
 int v;
 for (;;) {
  if ((c = pnm_getc(in)) == (-1)) {
   return -1;
  }
  if (c == '#') {
   for (;;) {
    if ((c = pnm_getc(in)) == (-1)) {
     return -1;
    }
    if (c == '\n') {
     break;
    }
   }
  } else if (c == '0' || c == '1') {
   v = c - '0';
   break;
  }
 }
 if (val) {
  *val = v;
 }
 return 0;
}
