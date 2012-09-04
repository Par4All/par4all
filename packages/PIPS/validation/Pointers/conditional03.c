/* A variation of conditional01.c */

char * x="a", *y="b", *z="c";

char * conditional03(int i)
{
  char * p[3] = {x, y, z};
  char * r = (i<0||i>2)? p[0]:p[i];
  return r;
}
