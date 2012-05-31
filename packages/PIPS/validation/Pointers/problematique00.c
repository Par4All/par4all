void read(int d){
  ;
}
int main(){
     int *a, b, c, d ;
     c = 3;       /* S1 */
     a = &c;      /* S2 */
     read(d);     /* S3 */
     b = *a;       /* S4 */


     return 0;
}
