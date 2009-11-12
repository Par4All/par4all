typedef enum {TRUE, FALSE} bool;

typedef struct allocatable_2d {
bool allocated;
int min1;
int max1;
int min2;
int max2;
double (*array)[max1-min1+1][max2-min2+2]; // ca passe?!?
} allocatable_2d;

main()
{
  allocatable_2d a;
}
