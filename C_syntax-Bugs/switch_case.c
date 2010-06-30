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
  default:
    break;
  }
  return 0;
}
