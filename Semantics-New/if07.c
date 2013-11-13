// After all if, j must be equal to 0

void if07()
{
  int i, j=-1, k;
  
  i=k;

  if(i==k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i!=k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i<=k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i>=k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i<k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i>k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  k=0;
  i=k;
  
  j=-1;
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
  
  
  j=-1;
  if(i==k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;

  j=-1;
  if(i!=k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i<=k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i>=k)
    // Should be reached
    j = 0;
  else
    // Should not be reached
    j = 1;
  
  j=-1;
  if(i<k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  j=-1;
  if(i>k)
    // Should not be reached
    j = 1;
  else
    // Should be reached
    j = 0;
  
  
  return;
}

int main()
{
  if07();
  return 0;
}

