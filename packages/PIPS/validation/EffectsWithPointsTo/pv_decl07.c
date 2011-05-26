// aggregate structure with pointers declaration
typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;
int main()
{
  mystruct s;
  return(0);
}
