#include <string.h>
typedef struct {
    int width, height;
} t;
void using_memcpy(int nb, t resC[nb], unsigned char* res[nb], unsigned char* src[nb])
{
    for (int i = 0; i < nb; ++i)
    {
        int size = resC[i].width * resC[i].height * sizeof(unsigned char);
        memcpy(res[i], src[i], size);
    }
}
