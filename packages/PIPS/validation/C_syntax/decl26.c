/* union with a structure inside used to declare a global variable */
union {
  int buf;
  struct{
    int empty : (32 - (7 + 9));
    char y : 7 ;
    short int x : 9;
  }w;
}adr;
