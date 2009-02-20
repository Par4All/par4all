int
find_while(int i) {

 begin2:
  if (i++ < 9)
    goto end2;
  goto begin2;

 end2:
  i++;
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
