int main() {
  int i;
  
#pragma if1
  if (i==0) {
    i++;
  }
  else {
    i--;
  }
#pragma X
    i=0;
    
#pragma if2
  if (i==0) {
    i++;
  }
  else 
    i--;
#pragma X
    i=0;
    
#pragma if3
  if (i==0) 
    i++;
  else {
    i--;
  }
#pragma X
    i=0;
    
#pragma if4
  if (i==0) 
    i++;
  else 
    i--;
#pragma X
    i=0;
    
    
  return i;
}
