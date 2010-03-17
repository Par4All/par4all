/* See what happens with multiple parameter declarations on one line and
   unordered declarations. This is OK with "gcc -c -ansi". */

int csplit11(i, pj, k, pl)
     int i, k;
     int * pl;
     int * pj;
{
  *pj = i;
  *pl = k;
}
