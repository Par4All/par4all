// dereferencing a null pointer yields a user error

// stdio.h is not included
#define NULL (0)

int main()
{
  int **a;
  int b;
  a = (int **) NULL;
  a[0] = &b;
  return(0);
}
