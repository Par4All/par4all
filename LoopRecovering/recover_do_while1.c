int
find_do_while(int i) {

 begin:
  i = 8*i;
  if (i++) {
    goto begin;
  }
  else
    i--;

 begin2:
  i = 9+i;
  if (i < 9)
    goto begin2;

  return i;
}
