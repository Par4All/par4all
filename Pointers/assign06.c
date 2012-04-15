/* To check list_assignment_to_points_to() */

void assign06()
{
  int ** ppi, *pi, *qi, i, j, k, l, m;
  ppi = i==0? &pi : &qi;
  pi = i==0? &i : &j;
  qi = i==0? &k : &l;
  *ppi = &m;
  return;
}
