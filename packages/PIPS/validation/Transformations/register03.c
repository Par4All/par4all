int main(void)
{
  // check declaration splitting
  int i0, i1, i2, i3, i4, i5, i6, i7, i8, i9;
  int * p;
  i0 = 0;
  i1 = 1;
  i2 = 2;
  i3 = 3;
  i4 = 4;
  i5 = 5;
  i6 = 6;
  i7 = 7;
  i8 = 8;
  i9 = 9;
  // break some register candidates
  p = &i1;
  p = &i3;
  p = &i5;
  p = &i7;
  p = &i9;
  return i0+i1+i2+i3+i4+i5+i6+i7+i8+i9;
}
