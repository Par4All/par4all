
main()
{
  typedef int myint;
  myint n;
  typedef myint MYINT; 
  MYINT m;
  struct s {
    int i;
  };
  typedef struct s mystruct;
  typedef union v {
    int i;
  } myunion;
  typedef myunion MYUNION;
  typedef MYUNION toto;
  typedef mystruct MYSTRUCT;
  typedef toto tata;
  typedef tata titi;
  n = 1;
}
