#include <math.h>
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif


#ifdef __cplusplus
extern "C" {
#endif

double get_time();
void timer_start();
void timer_stop();
void timer_display();
void timer_stop_display();

#ifdef __cplusplus
}
#endif
