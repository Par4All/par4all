
void Run( double a[10][10], int *p) {
  {
    int lv1, lv2;
    for (lv1 = 0; lv1 < 10; lv1++) {
      for (lv2 = 0; lv2 < 10; lv2++) {
        a[lv1][lv2] = (double)(1.0);
      }
    }
  };

  {
// This anywhere effect prevent lv2 privatisation in lv1 loop, 
// (but not the parallelization of lv2 loop) !
    int b = *p;
  };
  
  
}

