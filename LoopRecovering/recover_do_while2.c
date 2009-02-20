int
find_do_while(int i) {
 begin:
  i = 8*i;
  i++;
  if (i < 9) {
    goto the_end;
  }
  goto begin;

 the_end:
  return i;
}
