int rand(float z)
{
  return (int) z;
}

void function_name_conflict01(int x)
{
  float y;

  y = 1.;
  x = rand(y);
}
