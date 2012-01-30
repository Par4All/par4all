/* (c) HPC Project - 2010-2011 */

#define COLD_MODE               (1 << 0)
#define COLD_MODE_STANDALONE    0
#define COLD_MODE_JNI           1


void scilab_rt_init(int argc, char* argv[], int mode);

void scilab_rt_init2(int argc, char* argv[], int mode, void (*code)() );
