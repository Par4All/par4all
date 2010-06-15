int main()
{
  int i,j,k;
   int a[500];

   i = 0;
   j = 1;
   for (i=0; i<=499; i++) {
     j++;
     // Cumulated effects should be a[i] here and not a[*]
     a[i] = i;
   }
   k = 2;
   return k;
}
