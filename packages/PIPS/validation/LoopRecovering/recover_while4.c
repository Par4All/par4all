int
find_while(int i) {

 begin2:

 begin:
  if (i < 9) {
    i++;
    goto begin;
  }
  return i;
}
