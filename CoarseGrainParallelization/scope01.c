
void Run() {
  
  double a[10][10];
  {
    int lv1, lv2;
    for (lv1 = 0; lv1 < 10; lv1++) {
      for (lv2 = 0; lv2 < 10; lv2++) {
        a[lv1][lv2] = (double)(1.0);
      }
    }
  };
  double b;
  int lv0;

// This anywhere effect prevent lv2 privatisation, but not the parallelization !
  lv0 = *&b;
  
}

