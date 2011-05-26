// Debug the destrucuration of a for loop by the new controlizer

void for01()
{
  int c;
  int k = 0;
  for(c = 10; c>0; c--, k++) {
    if(c % 2 == 0) goto end;
  }
  c++;
 end:
  return;
}
