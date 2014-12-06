//check management of the return

int main() {
  int i;
  
#pragma distributed on_cluster=0
  {
    i=0;
    i++;
  }
  
  return i;
}

