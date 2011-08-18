// assignment of aggregate structure

// stdio.h is not included
#define NULL ((void *) 0)

typedef struct MY_LIST {int n; struct MY_LIST * next;} my_list;
int main()
{
  my_list l1, l2, l3, l4;
  my_list *l;

  l1.n = 1;
  l2.n = 2;
  l3.n = 3;
  l4.n = 4;
  l1.next = &l2;
  l2.next = (my_list *) NULL;
  l3.next = (my_list *) NULL;
  l4.next = (my_list *) NULL;

  l = &l1;
  l = l->next;
  *l = l3;
  l = &l4;
  l3.next = (my_list *) NULL;
  return(0);
}
