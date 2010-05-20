
void myKernel(int a[100], int b[100]){
  int i ;
  for(i = 50 ; i < 100; i++){
    b[i]=a[i];
  }
}

int main(){

  int a[100]={0};
  int b[100]={1};

  myKernel(a,b);

  return 0;
}
