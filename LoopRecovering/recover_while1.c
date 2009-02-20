int
find_while(int i) {
 begin:
  if (i++) {
    i = 8*i;
    goto begin;
  }
  else
    i--;

 begin3:
  if (i++ < 9)
    goto begin3;

 begin2:
  if (i < 9) {
    i = 8*i;
    goto begin2;
  }
  return i;
}
