int main()
{
  char c;
  switch(c) {
  case 'a':
    c++;
    break;
  case 'b':
    c--;
    break;
  case '?':
    c=0;
    break;
  case '\n':
    c = 1;
    break;
  case '\001':
    c = '\001';
    break;
  case '\002':
    c = '\002';
    break;
  case '\177':
    c = '\177';
    break;
  case '\277':
    c = '\277';
    break;
  case '\377':
    c = '\377';
    break;
  default:
    break;
  }
  return 0;
}
