//declaration of variable with same name inside a scope

int main() {
  int i;
  
#pragma distributed on_cluster=0
  {
    i=0;
  }
#pragma distributed on_cluster=1
  {
    int i;
    i=42;
  }
  
  //i=0
  return i;
}

