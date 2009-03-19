int
find_while(int i) {

 begin2:
  if (i++ < 9)
    goto end2;
  goto begin2;

  // Just verify it works with 2 consecutive labels:
 end2:

 begin:
  if (i < 9) {
    i++;
    goto the_end;
  }
  i = 8*i;
  goto begin;

 the_end:
  return i;
}
