char * conditional01(int i)
{
  char * p[3] = {"a", "b", "c"};
  return (i<0||i>2)? p[0]:p[i];
}
