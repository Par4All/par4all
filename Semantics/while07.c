int main()
{
  int i,j,k;
   int a[500];

   i = 0;
   j = 1;
   while (i<=499) {
     // The lower bound on i and j should not be lost...
     j++;
     // Cumulated effects should be a[i] here and not a[*]
     a[i] = i;
     i = i+1;
   }
   // We should have the precondition i == 500, j == 501
   // if we would use the proper fix point operator (derivative)
   k = 2;
   return k;
}
