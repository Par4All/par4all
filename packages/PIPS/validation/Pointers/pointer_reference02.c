/* To check pointer references */

int pointer_reference02(char **p)
{
  char * q = p[3];
  char * r;
  r = p[3];
  return q-p[3];
}
