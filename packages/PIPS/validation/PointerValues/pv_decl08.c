// nested aggregate structures with pointers declaration
typedef struct {int *a; int *b[10]; int (*c)[10];} myfirst_struct;
typedef struct {
  myfirst_struct fs; 
  myfirst_struct fs_t[10]; 
  myfirst_struct * fs_p; 
  myfirst_struct *fs_tp[10]; 
  myfirst_struct (* fs_pt)[10];
} mysecond_struct;
int main()
{
  mysecond_struct s;
  return(0);
}
