
int main()
{
  int i = 0;
  int *p, *q;
  int **pp;
  
  p = &i;
  pp = &p;
  q = *pp;      //On veut avoir p=q
  q = p;
  
  //on modifie i
  *q = 1;
  *p=2;
  **pp = 3;
  
  return 0;
}
