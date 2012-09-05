int pointer01(void)
{
  int vol, bla;
  int * p;
  int * q = &bla;
  int * r = &vol;
  r = &bla;
  p = &bla;
  vol = 12;
  *p = vol;
  *q = vol+1;
  *r = vol+2;
  return bla;
}
