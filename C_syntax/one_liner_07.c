/* In fact, check that partially empty for control are properly parsed and prettyprinted */

void one_liner_07()
{
  int x,y,z;
  for(;;)
    y++;
  for(x=0;;)
    y++;
  for(;x<10;)
    y++;
  for(;;x++)
    y++;
  for(x=0;x<10;)
    y++;
  for(x=0;;x++)
    y++;
  for(;x<10;x++) {
    y++;
  }
  for(;;)
    ;
}
