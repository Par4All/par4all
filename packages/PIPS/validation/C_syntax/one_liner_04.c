void one_liner_04()
{
  int x,y,z;

  if(x>0)
    y++;
  else
    y++;

  if(x>0) {
    y++;
    z++;
  }
  else {
    y++;
    z++;
  }

  if(x>0)
    ;
  else
    //comment empty branch, misplaced by the parser
    ;

  if(x>0)
    //simple comment, misplaced by the parser
    ;

  //last instruction
  x++;
  //last comment lost by the parser
}
