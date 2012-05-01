int cse_wpt01()
{
   // BEGIN BLOCK
   int i;
   int j;
   int k;
   int *p = &j;
   int *q = &k;
   i = 2*(j+2);
   *p = *q;
   k = 3*(j+2);
   // END BLOCK
}
