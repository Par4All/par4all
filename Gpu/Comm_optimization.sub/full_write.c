

void kernel(int a[10]) {
  for(int i=0; i<10;i++) {
    a[i]=0;
  } 
}

int main() {
  int a[10];
  
  a[0] = 0;

  // a is fully written, thus no copy in is needed
  kernel(a);
  
  int i = a[0];
}
  
