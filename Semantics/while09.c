/* check new_complete_while_loop() for a special case found in
   Fulguro */

void while09()
{
  int i = 1;

  i = 2;

  while(0)
    ;

  i = 3;
}
