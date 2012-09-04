/* Example for pointer subscripts: 4 subscripts */

void pointer18(float ****p)
{
  float z = 0;

  p[1][2][3] = &z;

  return;
}
