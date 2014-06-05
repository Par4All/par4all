int main() {
  int i;
  
#pragma for1
  for (i=0; i<10; i++) {
    i++;
  }
#pragma X
  i=0;
  
#pragma for2
  for (i=0; i<10; i++) 
    i++;
#pragma X
  i=0;
  
  return i;
}
