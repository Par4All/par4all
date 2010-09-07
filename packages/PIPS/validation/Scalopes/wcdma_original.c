//Includes
#include	<stdio.h>
#include	<math.h>
#include	<stdlib.h>

//debug
//#define DEBUG_SYNCHRO
//#define DEBUG_DATA
//#define DEBUG_MEMORY
//#define DEBUG_INFO
//#define CHECK_RESULT

#ifdef DEBUG_SYNCHRO
#define pdebug_synchro(fmt, args...) printf(fmt, ##args)
#else
#define pdebug_synchro(fmt, args...) /* nothing */
#endif
#ifdef DEBUG_DATA
#define pdebug_data(fmt, args...) printf(fmt, ##args)
#else
#define pdebug_data(fmt, args...) /* nothing */
#endif
#ifdef DEBUG_MEMORY
#define pdebug_memory(fmt, args...) printf(fmt, ##args)
#else
#define pdebug_memory(fmt, args...) /* nothing */
#endif
#ifdef DEBUG_INFO
#define pdebug_info(fmt, args...) printf(fmt, ##args)
#else
#define pdebug_info(fmt, args...) /* nothing */
#endif



//Constants
#define true  					1
#define false 					0

//number of symbol by frame (nb bit by pixel)
#define NB_SYMBOL				8*sizeof(int32_t)
//size of code
#define SF						8
//code for user 1
#define OVSF_CODE_USER_1		{-1,-1,1,1,-1,-1,1,1}
//used by QPSK
#define SAMPLING_FACTOR			4
//filte coeff
#define FILTER_COEFF			{1,0,0,0,0,0,0,0}
//filter nb cell
#define FILTER_NB_CELL			8
//delay when multi-path
#define DELAY_MAX				6
//pilot bits
#define PILOTS					{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
//pilot frame rate	MUST BE EVEN!!!
#define PILOT_RATE				10


#define sizeX 512
#define sizeY 1024

//Main thread
int main()
{
					  
  unsigned char image_in[sizeX*sizeY];
  unsigned char image_out[sizeX*sizeY];
	
  int32_t img_index=0;
	
  //Local variables
  //loop counter
  int32_t i,j,d,c,h,u,n,t;
	
  //Sending...	
  //generate frame
  int32_t data=0, data_bit;
  int32_t symbole_flow_user1[NB_SYMBOL];
	
  //spreading
  int32_t ovsf_code_user1[SF] = OVSF_CODE_USER_1;	//user code
  int32_t pilot[NB_SYMBOL] = PILOTS;
  int32_t spreading_signal1[(SF*NB_SYMBOL+DELAY_MAX)];
  int32_t spreading_signals[(SF*NB_SYMBOL+DELAY_MAX)];
  int32_t channel_delay1=0,channel_delay2=0,channel_delay3=0,channel_delay4=0;
	
  //demux
  int32_t I_user[SF/2*NB_SYMBOL+DELAY_MAX/2];	
  int32_t Q_user[SF/2*NB_SYMBOL+DELAY_MAX/2];	
	
  //qpsk
  float Signal_I[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
  float Signal_Q[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	
  //filter
  float FIR_COEFF[FILTER_NB_CELL]=FILTER_COEFF;	//filter coeff
  float x_buffer_user1[FILTER_NB_CELL];//buffer for convolution
  int32_t ptr_x_buffer_user1;	//ptr on buffer
  float channel_I[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
  float channel_Q[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
  float FIR2_I_user1[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
  float FIR2_Q_user1[(SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR];
	
  //inv qpsk
  int32_t *destination;
  int32_t inv_qpsk_user1[SF*NB_SYMBOL+DELAY_MAX];
  int32_t R_pilot[SF*NB_SYMBOL+DELAY_MAX];
	
  //channel estimation
  int32_t retro_loop_count=0;
  int32_t finger_mat [NB_SYMBOL][DELAY_MAX];
  int32_t coeff [DELAY_MAX];
  int32_t coeff2 [DELAY_MAX];//for the temporary coeff calculated before retroaction loop
  int32_t max;
  int32_t S_pilot[NB_SYMBOL] = PILOTS;	//sent pilots
	
  //rake receiver
  int32_t finger[DELAY_MAX];
  int32_t fingers;
  int32_t symbole_flow[NB_SYMBOL];
	
  int32_t found_error;
	
  //for read & write
  int32_t write;
  int flag = 1;
  int j_bkp;
  unsigned char init = 0;
  srand (0);
  //init the input image
  for(i=0;i<sizeX*sizeY;i++)
    image_in[i]= init++;//(unsigned char) (rand()%256);
	
	
  //Loop on the number of frame. The number of frame is made of "nb data frame" + "nb pilot frame". 
  //Pilot frames are added to initial frames for the channel estimation.
  for(t=0;t<(sizeX*sizeY)/sizeof(int32_t) + (sizeX*sizeY)/sizeof(int32_t)/(PILOT_RATE-1)+1;t++) 
    {
      pdebug_synchro("\n\nframe %i\n",t);
		
      //------------------------------------------ CREATE FRAME -----------------------------------------------
      //normal frame
      {int task;
#pragma scmp task
	task=0;
	if (t%PILOT_RATE!=0)
	  {
	    data = image_in[img_index]<<0 | image_in[img_index+1]<<8 | image_in[img_index+2]<<16 | image_in[img_index+3]<<24; //put 4 pixels in a frame (data)
	    pdebug_synchro("send signal\n");
	    pdebug_data("%#X\n",data);
	    for(i=0;i<NB_SYMBOL;i++) {
	      data_bit=(data>>i)&0x01;
	      symbole_flow_user1[i]=data_bit;
	      pdebug_data("%i ",data_bit);
	    }
	    pdebug_data("\n");
			
	    //---- Spreading
	    pdebug_synchro("spreading\n");
			
	    for(i=0; i<NB_SYMBOL; i++) {
	      for(j=0; j<SF; j++) {
		if(symbole_flow_user1[i]==0) spreading_signal1[i*SF+j] = -ovsf_code_user1[j];
		else spreading_signal1[i*SF+j] = ovsf_code_user1[j];
		pdebug_data("%i ",spreading_signal1[i*SF+j]);
	      }
	      pdebug_data("\n");
	    }
	  }
	//pilot frame
	else{
	  pdebug_synchro("send pilot\n");
			
	  /* pdebug_info("main path delay: 0\n"); */
	  channel_delay1=(int32_t)((rand()%DELAY_MAX));
	  /* pdebug_info("path 1 delay: %i\n",channel_delay1); */
	  channel_delay2=(int32_t)((rand()%DELAY_MAX));
	  /* pdebug_info("path 2 delay: %i\n",channel_delay2); */
	  channel_delay3=(int32_t)((rand()%DELAY_MAX));
	  /* pdebug_info("path 3 delay: %i\n",channel_delay3); */
	  channel_delay4=(int32_t)((rand()%DELAY_MAX));
	  /* pdebug_info("path 4 delay: %i\n",channel_delay4); */
	 //printf("%d %d %d %d\n",channel_delay1,channel_delay2,channel_delay3,channel_delay4);		
	  for(i=0; i<NB_SYMBOL; i++) {
	    write= pilot[i]==0?-1:1;
	    for(j=0; j<SF; j++) {
	      spreading_signal1[i*SF+j]=write;
	       //printf("%d",spreading_signal1[i*SF+j]);
	    }
	  }
	  //printf("\n");
	}
      }
      //generate multipath
      //main path
#pragma scmp task
      for(i=0;i<(SF*NB_SYMBOL);i++) spreading_signals[i]=spreading_signal1[i];
      
      for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf("%d ",spreading_signals[i]);
      }
      //printf("\n");

      //initialisation
#pragma scmp task
      for(i=SF*NB_SYMBOL;i<(SF*NB_SYMBOL+DELAY_MAX);i++) spreading_signals[i]=0;

      for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf(" | %d ",spreading_signals[i]);
      }
      //printf("\n");


      //other path
#pragma scmp task
      for(i=channel_delay1;i<(SF*NB_SYMBOL)+channel_delay1;i++) spreading_signals[i]+=spreading_signal1[i-channel_delay1];

for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf("%d ",spreading_signals[i]);
      }
      //printf("\n");

#pragma scmp task
      for(i=channel_delay2;i<(SF*NB_SYMBOL)+channel_delay2;i++) spreading_signals[i]+=spreading_signal1[i-channel_delay2];

for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf("%d ",spreading_signals[i]);
      }
      //printf("\n");

#pragma scmp task
      for(i=channel_delay3;i<(SF*NB_SYMBOL)+channel_delay3;i++) spreading_signals[i]+=spreading_signal1[i-channel_delay3];
for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf("%d ",spreading_signals[i]);
      }
      //printf("\n");



#pragma scmp task
      for(i=channel_delay4;i<(SF*NB_SYMBOL)+channel_delay4;i++) spreading_signals[i]+=spreading_signal1[i-channel_delay4];

for(i=0;i<(SF*NB_SYMBOL+DELAY_MAX);i++){
	//printf("%d ",spreading_signals[i]);
      }
      //printf("\n");

		
		
      //------------------------------------------ SEND FRAME -----------------------------------------------
      //---- Conversion serial to parallel
      pdebug_synchro("Conversion serial to parallel\n");
#pragma scmp task
      for(i=0,j=0; i<(SF*NB_SYMBOL+DELAY_MAX); i+=2,j++)
	{
	  I_user[j] = spreading_signals[i];
	  pdebug_data(" (%d / %d)", I_user[j],j);
	  Q_user[j] = spreading_signals[i+1];
	}
	pdebug_data("\n");	
      //---- Modulation QPSK on I
      pdebug_synchro("Modulation QPSK on I\n");
#pragma scmp task
      for(i=0; i<(SF/2*NB_SYMBOL+DELAY_MAX/2); i++) {
	//Phase = Sign of I_user
	for(h=0; h<SAMPLING_FACTOR; h++){
	  Signal_I[i*SAMPLING_FACTOR+h]=1.0*I_user[i]*cos((h*2*M_PI)/(SAMPLING_FACTOR-1));
	  pdebug_data("signal_I %f ",Signal_I[i*SAMPLING_FACTOR+h]);
	}
	pdebug_data("\n");
      }
		
      //---- Modulation QPSK on Q -- we suppose that we dephase it after of PI/2
      pdebug_synchro("Modulation QPSK on Q\n");
#pragma scmp task
      for(i=0; i<(SF*NB_SYMBOL/2+DELAY_MAX/2); i++)
	{
	  //Phase = Sign of Q_user
	  for(h=0; h<SAMPLING_FACTOR; h++){
	    Signal_Q[i*SAMPLING_FACTOR+h]=1.0*Q_user[i]*cos((h*2*M_PI)/(SAMPLING_FACTOR-1));
	    pdebug_data("%f ",Signal_Q[i*SAMPLING_FACTOR+h]);
	  }
	  pdebug_data("\n");
	}
		
		
      //---- FIR for I and Q before channel
		
      pdebug_synchro("FIR1 Filter in progress...\n");
		
      pdebug_synchro("FIR1 Filter on I\n");
      //initialisation
#pragma scmp task
      for(u=0;u<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);u++) channel_I[u] = 0;
      {int task;
#pragma scmp task
	task=0;
	for(u=0;u<FILTER_NB_CELL;u++) x_buffer_user1[u]=0;
	ptr_x_buffer_user1=0;
      }
		
      //convolution par la réponse impultionelle du filtre (FIR_COEFF)
#pragma scmp task
      for(i=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i++) {
	x_buffer_user1[ptr_x_buffer_user1++] = Signal_I[i];
	ptr_x_buffer_user1 %= FILTER_NB_CELL;
	for(n = (FILTER_NB_CELL-1) ; n >= 0 ; n-- ) {
	  channel_I[i] += FIR_COEFF[n] * x_buffer_user1[ptr_x_buffer_user1++];
	  ptr_x_buffer_user1 %= FILTER_NB_CELL;
	}
	pdebug_data("%f ",FIR2_I_user1[i]);
      }
      pdebug_data("\n");
		
      pdebug_synchro("FIR1 Filter on Q\n");
      //initialisation
#pragma scmp task
      for(u=0;u<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);u++) channel_Q[u] = 0;
      {int task;
#pragma scmp task
	task=0;		
	for(u=0;u<FILTER_NB_CELL;u++) x_buffer_user1[u]=0;
	ptr_x_buffer_user1=0;
      }	
#pragma scmp task
      for(i=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i++) {
	x_buffer_user1[ptr_x_buffer_user1++] = Signal_Q[i];
	ptr_x_buffer_user1 %= FILTER_NB_CELL;
	for(n = (FILTER_NB_CELL-1) ; n >= 0 ; n-- ) {
	  channel_Q[i] += FIR_COEFF[n] * x_buffer_user1[ptr_x_buffer_user1++];
	  ptr_x_buffer_user1 %= FILTER_NB_CELL;
	}
	pdebug_data("%f ",FIR2_Q_user1[i]);
      }
      pdebug_data("\n");
		
		
		
      //------------------------------------------ RECEPT FRAME -----------------------------------------------
		
      //---- FIR for I and Q after channel
		
      pdebug_synchro("FIR2 Filter in progress...\n");
#pragma scmp task		
      for(u=0;u<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);u++) FIR2_I_user1[u]=0;
#pragma scmp task
      for(u=0;u<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);u++) FIR2_Q_user1[u]=0;
		
      pdebug_synchro("FIR2 Filter on I\n");
      {int task;
#pragma scmp task
	task=0;
	for(u=0;u<FILTER_NB_CELL;u++) x_buffer_user1[u]=0;
	ptr_x_buffer_user1=0;
      }
      //convolution par la réponse impultionelle du filtre (FIR_COEFF)
#pragma scmp task
      for(i=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i++) {
	x_buffer_user1[ptr_x_buffer_user1++] = channel_I[i];
	ptr_x_buffer_user1 %= FILTER_NB_CELL;
	for(n = (FILTER_NB_CELL-1) ; n >= 0 ; n-- ) {
	  FIR2_I_user1[i] += FIR_COEFF[n] * x_buffer_user1[ptr_x_buffer_user1++];
	  ptr_x_buffer_user1 %= FILTER_NB_CELL;
	}
	pdebug_data("%f ",FIR2_I_user1[i]);
      }
      pdebug_data("\n");
		
      pdebug_synchro("FIR2 Filter on Q\n");
      {int task;
#pragma scmp task
	task=0;
	for(u=0;u<FILTER_NB_CELL;u++) x_buffer_user1[u]=0;
	ptr_x_buffer_user1=0;
      }
#pragma scmp task	
      for(i=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i++) {
	x_buffer_user1[ptr_x_buffer_user1++] = channel_Q[i];
	ptr_x_buffer_user1 %= FILTER_NB_CELL;
	for(n = (FILTER_NB_CELL-1) ; n >= 0 ; n-- ) {
	  FIR2_Q_user1[i] += FIR_COEFF[n] * x_buffer_user1[ptr_x_buffer_user1++];
	  ptr_x_buffer_user1 %= FILTER_NB_CELL;
	}
	pdebug_data("%f ",FIR2_Q_user1[i]);
      }
      pdebug_data("\n");
		
		
		
      //---- QPSK-1 (+ parallel to serial)
      pdebug_synchro("QPSK-1 in progress...\n");
      /*I have remove the pointer destination*/
      {int task;
#pragma scmp task
	task=0;
	if (t%PILOT_RATE==0){
	  pdebug_synchro("receive pilot\n");
	  for(i=0,j=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i+=SAMPLING_FACTOR,j+=2) {
	    //It is a cosinus, so the first bit out of 4 sampled gives
	    //the sign of the symbol and the amplitude of the signal.
	  R_pilot[j]=(int32_t)FIR2_I_user1[i];
	  pdebug_data("%i ",inv_qpsk_user1[j]);
	  R_pilot[j+1]=(int32_t)FIR2_Q_user1[i];
	  pdebug_data("%i ",inv_qpsk_user1[j+1]);
	  }
	}
	else{
	  for(i=0,j=0;i<((SF/2*NB_SYMBOL+DELAY_MAX/2)*SAMPLING_FACTOR);i+=SAMPLING_FACTOR,j+=2) {
	    //It is a cosinus, so the first bit out of 4 sampled gives
	    //the sign of the symbol and the amplitude of the signal.
	    inv_qpsk_user1[j]=(int32_t)FIR2_I_user1[i];
	    pdebug_data("%i ",inv_qpsk_user1[j]);
	    inv_qpsk_user1[j+1]=(int32_t)FIR2_Q_user1[i];
	    pdebug_data("%i ",inv_qpsk_user1[j+1]);
	  }
	}
      }
      pdebug_data("\n");	
      //---- Channel estimation
      /*I have remove the goto*/
      {int task, j;
#pragma scmp task
	task=0;
      if(t%PILOT_RATE==0)
	{
	  //initialisation
	  for(j=0;j<DELAY_MAX;j++)
	    {
	      coeff[j]=0;
	      coeff2[j]=0;
	    }

	  retro_loop_count=0;
	  flag=1;	
	  while(flag==1){
	    //estimation:
	    flag=0;
			
	    //create matrix
	    for(c=0;c<NB_SYMBOL;c++)
	      {
		for(i=0;i<DELAY_MAX;i++) finger_mat[c][i]=0;	//initialisation
		for(j=0;j<DELAY_MAX;j++)
		  {
		    //for each finger... (code on the signal shifted to the right)
		    for(d=0;d<SF;d++) finger_mat[c][j]+=R_pilot[c*SF+d+j];
		  }
	      }
			
	    //calculate the channel coefficients
	    for(j=0;j<DELAY_MAX;j++)
	      {
		coeff2[j]=0;
		for(c=0;c<NB_SYMBOL;c++)
		  {
		    if(S_pilot[c]==0)
		      {
			coeff2[j]+= finger_mat[c][j]*-1;
		      }
		    else
		      {
			coeff2[j]+= finger_mat[c][j]*1;
		      }
		  }
	      }

	    //calculating max power received
	    max=0;
	    for(j=0;j<DELAY_MAX;j++)
	      {
		if(coeff2[j]>max)
		  max=coeff2[j];
	      }
				
	    if(max>=(SF*NB_SYMBOL))
	      {

		//retroaction
		for(j=0;j<DELAY_MAX;j++)
		  {

		    if(retro_loop_count>DELAY_MAX){j_bkp = j; j=DELAY_MAX; }//break;
		    else{
		    if(coeff2[j]>=max)
		      {
			coeff[j]++;
			for(c=0;c<NB_SYMBOL;c++)
			  {
			    for(d=0;d<SF;d++)
			      {
				if(S_pilot[c]==1)
				  R_pilot[c*SF+d+j]-=1;
				else
				  R_pilot[c*SF+d+j]+=1;
			      }
			  }
			retro_loop_count++;
			flag=1;
			j_bkp = j;
			j=DELAY_MAX;
			//goto estimation;
		      }}
		  }
		j= j_bkp;
	      }
	  }
	}	
      }
      //---- Rake receiver
      {int task;
#pragma scmp task
	task=0;
	if(t%PILOT_RATE!=0)
	  {
	    pdebug_synchro("Rake Receiver in progress...\n");
	    data=0;
	    for(c=0;c<NB_SYMBOL;c++) 
	      {
		fingers=0;
		for(i=0;i<DELAY_MAX;i++) finger[i]=0;
		for(j=0;j<DELAY_MAX;j++) {
		  //for each finger... (code on the signal shifted to the right)
		  for(d=0;d<SF;d++){
		    finger[j]+=inv_qpsk_user1[c*SF+d+j]*ovsf_code_user1[d];
		    //printf("%d %d %d ",finger[j],inv_qpsk_user1[c*SF+d+j],ovsf_code_user1[d]);
		  }
		}
		//cout<<endl;
		for(i=0;i<DELAY_MAX;i++) {
		  fingers+=coeff[i]*finger[i];
		}
		if(fingers>0) data_bit=1;
		else data_bit=0;
		data=data|data_bit<<c;
		symbole_flow[c]=data_bit;
	      }
	  }

      }	

	//---- Store result
	if(t%PILOT_RATE!=0)
	  {
	    //store received frame in image_out
	    image_out[img_index] =   (data & 0x000000FF) >> 0;
	    image_out[img_index+1] = (data & 0x0000FF00) >> 8;
	    image_out[img_index+2] = (data & 0x00FF0000) >> 16;
	    image_out[img_index+3] = (data & 0xFF000000) >> 24;
			
	    img_index+=4;	//for next frame, increment index by size of frame
			
	    pdebug_data("\nReceived signal =\n");
	    for(d=0;d<NB_SYMBOL;d++) {
	      pdebug_data("%i ",symbole_flow[d]);
	    }
	    pdebug_data("\n");
	  }
#ifdef CHECK_RESULT
	found_error=false;
	for(d=0;d<NB_SYMBOL;d++) {
	  if((symbole_flow_user1[d]-symbole_flow[d])!=0) {
	    pdebug_info("error detected with symbole_flow_user1[%d]= %d \t symbole_flow[%d]= %d\n",d,symbole_flow_user1[d],d,symbole_flow[d]);
	    found_error=true;
	  }
	}
	if(!found_error){
	  pdebug_info("check complete!\n");
	}
	else {
	  //printf("ERROR in output image\n");
	  exit(EXIT_FAILURE);
	}
#endif
	



    }
  //Debug
  /* pdebug_memory("input image\n"); */
  /* for(i=0;i<sizeY;i++){ */
  /*   for(j=0;j<sizeX;j++){ */
  /*     pdebug_memory("%d\t",image_in[i*sizeX+j]); */
  /*   } */
  /*   pdebug_memory("\n"); */
  /* } */
	
  /* pdebug_memory("output image\n"); */
  /* for(i=0;i<sizeY;i++){ */
  /*   for(j=0;j<sizeX;j++){ */
  /*     pdebug_memory("%d\t",image_out[i*sizeX+j]); */
  /*   } */
 /*   pdebug_memory("\n"); */
  /* } */
	
  //WARNING NOT TO BE PORTED TO SESAM just to check result with diff
 /*  FILE * output_file= fopen("output_image","w"); */
/*   if(output_file==NULL){ printf("ERROR when opening output file\n");exit(EXIT_FAILURE);} */
/*   fwrite(image_out, sizeof(unsigned char), sizeX*sizeY,output_file); */
/*   fclose(output_file); */
	
/*   FILE * input_file= fopen("input_image","w"); */
/*   if(input_file==NULL){ printf("ERROR when opening input file\n");exit(EXIT_FAILURE);} */
/*   fwrite(image_in, sizeof(unsigned char), sizeX*sizeY,input_file); */
/*   fclose(input_file); */
	
  return(EXIT_SUCCESS);
}
