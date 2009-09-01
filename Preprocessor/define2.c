
#define INDICE(x,y) (x+y)
#define INDICE2(x,y) ((x)+(y))

main()
{
  int tmp;
  tmp = INDICE (+1,-1);
  tmp = INDICE (-1,+1);
  tmp = INDICE2(+1,-1);
  tmp = INDICE2(-1,+1);
  tmp = (+1 +-1);
  tmp = (-1 + +1);
  return tmp;
}
