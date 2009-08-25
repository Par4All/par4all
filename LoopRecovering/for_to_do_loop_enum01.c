enum {
  LB = 2,
  UB = 100,
  INC = 3
};

int main() {
  int a[100];
  int i, j;

  // A do loop ! Should be parallel
  for(i = LB; i < UB; i = i + INC)
     a[i] = 2;

  return 0;
}
