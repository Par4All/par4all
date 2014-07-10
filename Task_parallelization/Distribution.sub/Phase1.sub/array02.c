//dependence on the index of the array need to cmmunicate it

int main() {
  int a[10];
  int x, y;
  int r;
  
#pragma distributed on_cluster=2
  {
    int i;
    for (i=0; i<10; i++) {
      a[i] = 0;
    }
  }
#pragma distributed on_cluster=0
  {
    x = 5;
    y = 42;
    a[x] = y;
    x = -1;
    y = 24;
  }
  
#pragma distributed on_cluster=1
  {
    r = a[5] - 42;
  }
  
  //x=-1, y=24, a[5]=42, r=0
  return r;
}
