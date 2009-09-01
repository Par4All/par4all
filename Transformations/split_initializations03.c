/* Expected result: this declaration could/should be split, but is
   not for the time being.
   
   The second allocation (a = {...} seems to be accepted neither by
   gcc, nor by tpips. This is why it's commented out for now.
 */

void split_initializations03(void)
{
  typedef struct foo {
    int a;
    int b;
    int c;
  } three_ints;
  
  three_ints a = {1, 2, 3};
  //a = {4, 5, 6};
}
