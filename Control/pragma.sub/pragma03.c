int main() {
  int i;
  
#pragma toto
  {
    i=0;
    i++;
  }
#pragma X
  i=0;
  
  return i;
}

