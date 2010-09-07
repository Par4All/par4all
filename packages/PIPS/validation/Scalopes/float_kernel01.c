/*Test the handling of different var types in bufferization*/
int main(){
  int i;
  float a[100];
  int b[100];
  int j= 5;
#pragma scmp task
 for(i=0; i <100; i++){
    a[i]=i;
    b[i]=i;
    j+=i;
  }
}
