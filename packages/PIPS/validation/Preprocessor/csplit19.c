/* See if csplit does not declare too many types. */

typedef struct {
    int table ;
    float type ;
} set_chunk, *set ;

extern set set_add_element  ( set, set, void *)   ;


int csplit18()
{
  int i;

  /* table might be seen as a type by splitc.y */

  table x;

  i = 1;
}
