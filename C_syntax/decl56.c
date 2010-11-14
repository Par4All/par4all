/* Bug spotted by Mehdi: x should not be declared in the outer loops */

int main()
{
 int i, j, k;
 for (i = 0; i<  N; i++) {
   for (j = 0; j<  N; j++) {
     for (k = 0; k<  N; k++) {
       int x = 0;
       ;
     }
   }
 }
}
