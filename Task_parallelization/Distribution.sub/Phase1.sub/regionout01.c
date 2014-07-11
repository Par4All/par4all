//    T0    
// T1 | T2  can run parallele
//    T3
// cell value of a at the end:0,-1,0,1,0,1,0,1,-1,1
// with only region IN/OUT we can't run T1 and T2 parallele
// ie. we can't make optimisation between communication
// more than that some optimization can make the code false
//   -> with region IN/OUT, 
//      T1/T2 don't read a
//      T0 can think no need to communicate for T1 or T2
//      same for T1 with T2
//      so in the end, T3 will receive false data from T1 
//      and next after that also receive false data from T2
// in point of view of T3, with region, a risk to be: 
// -1,-1,-1,-1,-1,-1,-1,-1,-1,-1
// 0 , ?, 0, ?, 0, ?, 0,-1,-1,-1
// 0 , ?, 0, 1, ?, 1, ?, 1, ?, 1

#include <stdio.h>

int main() {
  int i;
  int a[10];
  
  //T0 init the array
#pragma distributed on_cluster=0
  for (i=0; i<10; i++) {
    a[i] = -1;
  }
  
  //T1 modify even cell without the last even cell (to be more tricky)
#pragma distributed on_cluster=1
  for (i=0; i<4; i++) {
    a[2i] = 0;
  }
  
  //T2 modify odd cell without the first odd cell (to be more tricky)
#pragma distributed on_cluster=2
  for (i=1; i<5; i++) {
    a[2i+1] = 1;
  }
  
  //T3 use all the array
#pragma distributed on_cluster=3
  for (i=0; i<10; i++) {
    printf("a[%d]=%d\n", i, a[i]);
  }
  
  //cell value of a at the end: 0,-1,0,1,0,1,0,1,-1,1
  return 0;
}