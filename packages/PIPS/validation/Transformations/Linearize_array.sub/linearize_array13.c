/**** use to break with :

  LINEARIZE_ARRAY                updating   CODE(linearize)
                                 updating   CODE(linearize_array13!)
                                 updating   ENTITIES()
pips error in intrinsic_call_to_type: (../../../../../src/pips/src/Libs/ri-util/type.c:2136) assertion failed

 'pointer arithmetic with array name, first element must be the address expression' not verified
***/

int linearize(int a[10][10],int b[10][10]) {
 int ret;
 
 ret = a[0][(int) (1+b[2][1])];
 
 return ret;
}

