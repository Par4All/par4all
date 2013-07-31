
int main()
{
  int i = 0;
  int *p, *q;
  int **pp;
  
  p = &i;
  pp = &p;
  //On veut avoir p=q
  q = *pp;
  // q = p;
  
  //on modifie i
  *q = 1;
  *p=2;
  **pp = 3;
  
  return 0;
}
