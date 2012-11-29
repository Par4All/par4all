// dereferencing an uninitialized pointer yields a user error

int main()
{
  int **a;
  int b;
  a[0] = &b; 
  return(0);
}
