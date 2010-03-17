/* Union with a structure inside used to declare a global variable
 *
 * Make sure that the entities are declared in the right order...
 */
union {
  int buf;
  struct{
    int x;
  } w;
} adr;
