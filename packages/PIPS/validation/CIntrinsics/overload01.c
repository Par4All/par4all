#include <string.h>

char * strdup(const char* s) {
    size_t len = strlen(s);
    char * new = malloc(sizeof(char)*(1+len));
    if(new) {
        strcpy(new,s);
    }
    return new;
}

/* strange redefinition ... */
static void MAX(int a[1]) {
    a[0]=0;
}
