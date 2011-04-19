int main(){
     int *p, *q, i, j;
     
     i = 1;
     j = 0;
/* Initialisation des pointeurs*/
     p = &i;
     q = &j;

/* Les deux pointeurs pointent vers la meme case memoire j */
     p = q;

/* Dereferencement du pointeur p -> acces à la variable j */
     *p = 1;

     return 0;
}
