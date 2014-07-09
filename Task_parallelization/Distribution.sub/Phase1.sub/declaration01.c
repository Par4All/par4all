//make declaration variable at many places in the code

int main() {
  int i;
  
#pragma distributed on_cluster=0
  {
    int x;
    x=0;
    i=x+1;
  }
  int x;
#pragma distributed on_cluster=1
  {
    x=10;
    i=i+x;
  }
  int j;
#pragma distributed on_cluster=0
  i++;
#pragma distributed on_cluster=1
  {
    int x;
    x=30;
    j=x;
  }
#pragma distributed on_cluster=0
  i=j+i;
  
  //i=42, x=10, j=30
  return i;
}

