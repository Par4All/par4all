int main(int toto) {
  int i;
  int a[10];
  int r;
  
  {
    i=0;
    i++;
  }
  
#pragma distributed on_cluster=0
  {
    a[i] = 42;
    i++;
  }
  
#pragma distributed on_cluster=1
  {
    i--;
    r = a[i] + i - 1; 
  }
  
  //i=1, a[1]=42, r=42
  return r;
}
