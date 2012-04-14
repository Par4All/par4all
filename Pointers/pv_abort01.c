// dereferencing an uninitialized pointer "a" yields a user error
int main()
{
  int **a, b;
  a[0] = &b; 
  return(0);
}
