// basic test statements with relational and logical operators 

int main()
{
  int *a;
  int b = 0;
  int c = 1;
  if ((b==c) || (b!=0))
    a = &b;
  else
    a = &c;

  if(b!=c)
    a = &b;
  else
    a = & b;
  return(0);
}
