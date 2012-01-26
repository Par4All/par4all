/* To check typedef depending on typedef */

void env04()
{
  typedef int mile;
  typedef mile km;
  km j;

  j = (km) 1;
}
