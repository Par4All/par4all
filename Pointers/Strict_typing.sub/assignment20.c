/* Debugging of list_assignment_to_points_to() ... */

// single array element assignments

int main()
{
  int *a[2], i, j;
  a[i] = &i;
  a[j] = &j;
  return(0);
}
