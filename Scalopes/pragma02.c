/*Test scalopragma on blocks*/
int main(){

 int a[100];
 int i;

 {
   int n;
   #pragma scmp task
   a[2]++;
   a[3]++;
 }

  return 0;
}
