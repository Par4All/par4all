/* The bug disappears if the label decorates "i=1;" instead of
   "{i=1;}"

   An assumption made in controlizer must be broken.
*/

int main()
{
  int i=0;
  if(i)
    {
    label:
      {
	i=1;
      }
    }
  if(!i)
    goto label;
  return 0;
}
