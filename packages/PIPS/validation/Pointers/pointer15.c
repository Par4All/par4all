// array of pointers towards arrays

void pointer15(double * (*t[3][4])[5][6][7])
{
  double z;
  (*(t[0][0]))[1][2][3] = &z;
  *(*(t[0][0]))[1][2][3] = 2.5;
  return;
}
