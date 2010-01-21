//Includes
#include "stdio.h"
#include "math.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
 

//Constants
//#define DEBUG
#define NB_SYMBOL		64
#define SF			16
//#define OVSF_CODE_USER_1	{-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1}
#define OVSF_CODE_USER_1 {1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1}
//#define OVSF_CODE_ref	{-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1}

//#define OVSF_CODE_ref	{-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1}
#define OVSF_CODE_ref	{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
#define SAMPLING_FACTOR		4
#define FILTER_NB_CELL		64
#define DELAY_MAX		16

//#define DEBUG

#undef M_PI
#define M_PI 3.14159265358979323846


//Main thread
int main()
{
  int i, d, j, c, h, u, n;
	//Local variables
	//Sending...	
	int symbole_flow_user1[NB_SYMBOL];
	int ovsf_code_user1[SF] = OVSF_CODE_USER_1;
	int ovsf_code_ref[SF] = OVSF_CODE_ref;
	int spreading_signal_user1[SF*NB_SYMBOL];
	int spreading_signals[SF*NB_SYMBOL+DELAY_MAX];
	
	int I_user[SF/2*NB_SYMBOL+DELAY_MAX/2];	
	int Q_user[SF/2*NB_SYMBOL+DELAY_MAX/2];	
	float Signal_I[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	float Signal_Q[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	float FIR_COEFF[FILTER_NB_CELL];
	
	//receiving
	float FIR2_I_user1[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	float FIR2_Q_user1[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	int inv_qpsk_user1[SF*NB_SYMBOL+DELAY_MAX];
	int symbole_flow[NB_SYMBOL];
	
	int count_loop=0;
	int max=0;
		
	/********************************************************************************/
	/*										*/
	/*										*/
	/********************************************************************************/
	while(count_loop<10)
	//while(true)
	{
	  int channel_delay1, channel_delay2, channel_delay3, channel_delay4;
	  float alpha = (float) (0.22);
		
		float x_buffer_user1[FILTER_NB_CELL];
		int ptr_x_buffer_user1=0;
		int found_error=0;
		int retro_loop_count=0;
		int finger_mat [NB_SYMBOL][DELAY_MAX];
		int coeff [DELAY_MAX];
		int coeff2 [DELAY_MAX];//for the temporary coeff calculated before retroaction loop
		int finger[DELAY_MAX];
		int fingers;
		
		count_loop++;

		estimation:
		
		//#ifdef DEBUG
		;
		for(d=0;d<(SF*NB_SYMBOL+DELAY_MAX);d++) {; ;}
		;
		//#endif
		
		
		for(c=0;c<NB_SYMBOL;c++)
		{
			for(i=0;i<DELAY_MAX;i++) finger_mat[c][i]=0;
			for(j=0;j<DELAY_MAX;j++)
			{
				for(d=0;d<SF;d++) finger_mat[c][j]+=inv_qpsk_user1[c*SF+d+j]*ovsf_code_ref[d];
			}
		}	
		;
		
		;
		for(c=0;c<NB_SYMBOL;c++)
		{
			;
			for(j=0;j<DELAY_MAX;j++)
			{
				if(finger_mat[c][j]>=0)
					printf("%3d\t",finger_mat[c][j]);
				if(finger_mat[c][j]<0)
					printf("%3d\t",finger_mat[c][j]);
				//if(j<DELAY_MAX-1);
			}
			;
		}
		;
		for(j=0;j<DELAY_MAX;j++)
		{
			coeff2[j]=0;
			for(c=0;c<NB_SYMBOL;c++)
			{
				if(symbole_flow_user1[c]==0)
				{
					coeff2[j]+= finger_mat[c][j]*-1;
				}
				else
				{
				coeff2[j]+= finger_mat[c][j]*1;
				}
			}
		}
		;
		
		//display the channel coefficients
		;
		for(j=0;j<DELAY_MAX;j++)
		{
			;
			if(j<DELAY_MAX-1);
		}
		;
		
		//calculating max power received
		max=0;
		for(j=0;j<DELAY_MAX;j++)
		{
			if(coeff2[j]>max)
				max=coeff2[j];
		}
		
		//max=max*100/(SF*NB_SYMBOL*80);
		
		;
		if(max>=(SF*NB_SYMBOL))
		{
		//retroaction
			for(j=0;j<DELAY_MAX;j++)
			{
				if(retro_loop_count>DELAY_MAX) break;
				if(coeff2[j]>=max)
				{
					coeff[j]++;
					for(c=0;c<NB_SYMBOL;c++)
					{
						for(d=0;d<SF;d++)
						{
							if(symbole_flow_user1[c]==1)
								inv_qpsk_user1[c*SF+d+j]-=ovsf_code_ref[d];
							else
								inv_qpsk_user1[c*SF+d+j]+=ovsf_code_ref[d];
						}
					}
					;
					
					retro_loop_count++;
					goto estimation;
				}	
			}
		}
	}
	;
	return(0);
}



