// Debug the destrucuration of a while loop by the new controlizer

void while01()
{
  int c = 10;
  while(c>0) {
    c--;
    if(c % 2 == 0) goto end;
  }
  c++;
 end:
  return;
}
