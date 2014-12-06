//simple sequece of instruction

int main() {
  int i;
  
#pragma distributed on_cluster=0 
  i=1;
#pragma distributed on_cluster=1 
  i++;
#pragma distributed on_cluster=0 
  {
    int x;
    x=i;
    i=2*x;
  }
  
  return i;
}

