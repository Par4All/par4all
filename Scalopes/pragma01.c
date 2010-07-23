/*Test Scalopragma on loops*/
int main(){

 int a[100];
 int i;

  #pragma scmp task
  for(i=0; i<100; i++){
      a[i]=i;
  }

  return 0;
}
