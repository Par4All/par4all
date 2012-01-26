/* To check sizeof, and use-def elimination */

int env05()
{
  typedef int mile;
  typedef mile km;
  km j;

  j = (km) 1;
  j = sizeof(km);
  return j;
}
