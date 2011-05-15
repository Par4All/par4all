// basic test statements with scalars
int main()
{
  int *a;
  int b = 0;
  int c = 1;
  if (1)
    a = &b;
  else
    a = &c;

  if(1)
    a = &b;
  else
    a = & b;
  return(0);
}
