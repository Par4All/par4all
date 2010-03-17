void one_liner_05()
{
  int x,y,z;
  if(x>0)
    y++;
  else if(x>0){
    //to please the parser
    y++;
  }
  if(x>0) {
    y++;
    z++;
  }
  else if(x>0) {
    y++;
    z++;
  }
  if(x>0)
     ;
  else if(x>0)
    ;
}
