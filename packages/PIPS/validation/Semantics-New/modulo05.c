// Bug in modulo_to_transformer, expended from modulo04

void modulo05(int *p)
{ 
  int i = 1;

  if((*p)++%2 !=0) {}

  return;
} 
