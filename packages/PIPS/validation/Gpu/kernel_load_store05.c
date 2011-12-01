

int kernel(int offset, int a[1000]) {
  a[offset]=0;
}

int main(int argc, char ** argv) { 
  int a[1000];
  int offset = argc;
  kernel(offset,a);

 
}
