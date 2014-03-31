// Check that arrays of struct are not a problem for split initialization

void split_initializations13()
{
  struct s {int one; float two;};
  struct s a[] = {{1, 2.},{1, 2.},{1, 2.}}; 

  a[0].one = 0;
  a[1].one = 1;
  a[2].one = 2;

  a[0].two = 0.;
  a[1].two = 1.;
  a[2].two = 2.;
}
