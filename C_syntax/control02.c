// Break in forever for

int control02(int *in, int *val)
{
 for (;;) {
     break;
 }
 if (val) {
  *val = *in;
 }
 return 0;
}
