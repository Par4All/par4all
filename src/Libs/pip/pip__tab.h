struct A
    {struct A *precedent;
     char *bout;
    };

struct L
    {int flags;
     union { int unit;
             Entier * val;
	   } objet;
    };

#define Unit 1
#define Plus 2
#define Minus 4
#define Zero 8
#define Critic 16
#define Unknown 32

#define Sign 62

#define Index(p,i,j) (p)->row[i].objet.val[j]
#define Flag(p,i)    (p)->row[i].flags
struct T
    {int height, width;
     struct L row[1];
    };

typedef struct T Tableau;

void tab_init();

char * tab_hwm();

void tab_reset();

Tableau * tab_alloc();

void tab_copy();

Tableau * tab_get();

void tab_display();

Tableau *tab_expand();
