
int main()
{
  int i, result = 0;
  int size = 10;
  int a[size];
  
  for (i=0; i<10; i++)
    a[i] = i;
  
  for (i=0; i<10; i++)
    result += a[i];
  
  return result;
}
