// After all if, j must be equal to 0

void if06()
{
  int i=0, j=-1;

  if(i==0)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;

  j=-1;
  if(i!=0)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i<=0)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i>=0)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i<0)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i>0)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  return;
}

int main()
{
  if06();
  return 0;
}

