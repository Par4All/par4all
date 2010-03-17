void nested_struct()
{
  typedef struct{
    int type;
    char name;
    union { float f; int i;} value;
    void *next;
  }  VARIABLE;
  
  VARIABLE v;
}
