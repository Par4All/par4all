// Initialization to nowhere of an array of struct

typedef struct {
  int *a;
  int *b[10];
  int (*c)[10];
} mystruct;

int main()
{
  mystruct tab_s[2];
  return(0);
}
