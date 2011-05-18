// dereferencing a null pointer yields a user error
int main()
{
  int **a;
  int b;
  a = (int **) NULL;
  a[0] = &b; 
  return(0);
}
