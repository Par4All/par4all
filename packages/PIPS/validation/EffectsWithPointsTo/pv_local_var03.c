// elimination of local array of aggregate structure containing pointers
typedef struct {int *a; int *b[10]; int (*c)[10];} mystruct;
int main()
{
  int i;
  mystruct s1, s2;

  i = 0;
  do
    {
      mystruct tab_s[2];
      tab_s[0] = s1;
      tab_s[1] = s1;
      tab_s[1] = s2;
      i++;
    } while (i<10);
  return(0);
}
