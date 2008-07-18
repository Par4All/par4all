// problem with comment placement
void one_liner_06()
{
  int x,y,z;
  /* first high-level test*/
  if(x>0)
    y++;
  else if(x>0){
    //to please the parser
    y++;
  }  x = 3;
  /* second high-level test*/
  if(x>0) {
    y++;
    z++;
  }
  else if(x>0) {
    y++;
    z++;
  }
  /* third high-level test*/
  if(x>0)
    ;
  else if(x>0)
    ;
  /* first for */
  for(x=0; x<10; x++)
    ;
  if(x>0)
    ;
  else
    /* first while */
    while(x>0)
      x--;
  if(x<0)
    /* first switch */
    switch(x) {
    default: break;
    }
}
