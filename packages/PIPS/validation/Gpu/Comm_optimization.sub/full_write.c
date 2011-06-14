

void kernel_full(int a[10]) {
  for(int i=0; i<10;i++) {
    a[i]=1;
  } 
}

void kernel_partial(int a[10]) {
  for(int i=0; i<10-1;i++) {
    a[i]=1;
  } 
}

int a[10];

int main() {
  // This variable is only here to include a preconditions an make it harder for us to clean regions ;-)
  int n = 1;

  a[0] = 0;
  int res;

  // a is fully written, thus no "copy-in" is needed
  kernel_full(a);

  res = a[0]; // Force a copy-out

  // Def on CPU
  a[0] = 0;

  // a is not fully written, thus a "copy-in" is needed
  kernel_partial(a);
  
  res = a[0]; // Force a copy-out
}
  
