int main() {
  int i;
  
#pragma toto
  {
    i=0;
#pragma tata
    i=0;
#pragma X
    i=0;
  }
#pragma X
  i=0;
  
  return i;
}

