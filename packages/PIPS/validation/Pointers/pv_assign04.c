// assignment of elements arrays of aggregate structure containing pointers 
typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;
int main()
{
  mystruct s1, s2;
  mystruct tab_s[2];
  tab_s[0] = s1;
  tab_s[1] = s1;
  tab_s[1] = s2;
  return(0);
}
