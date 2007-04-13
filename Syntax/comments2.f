                                                                        
C                                                                       
C FFTPACKAGE FOR PSEUDOSPECTRAL ADVECTION CALCULATIONS   05/04/84       
C                                                                       
C PURPOSE  THIS PACKAGE CONSISTS OF PROGRAMS WHICH PERFORM FAST FOURIER 
C          TRANSFORMS FOR BOTH COMPLEX AND REAL PERIODIC SEQUENCES AND  
C          CERTIAN OTHER SYMMETRIC SEQUENCES THAT ARE LISTED BELOW.     
C                                                                       
C            RFFTI     INITIALIZE  RFFTF AND RFFTB                      
C            RFFTF     FORWARD TRANSFORM OF A REAL PERIODIC SEQUENCE    
C            RFFTB     BACKWARD TRANSFORM OF A REAL COEFFICIENT ARRAY   
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTI(N,WSAVE)                                         
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN     
C     BOTH RFFTF AND RFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH  
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND      
C     STORED IN WSAVE.                                                  
C                                                                       
C     INPUT PARAMETER                                                   
C                                                                       
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.             
C                                                                       
C     OUTPUT PARAMETER                                                  
C                                                                       
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N+15.   
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH RFFTF AND RFFTB  
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS    
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF   
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF RFFTF OR RFFTB.
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTF(N,R,WSAVE)                                       
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL      
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED     
C     BELOW AT OUTPUT PARAMETER R.                                      
C                                                                       
C     INPUT PARAMETERS                                                  
C                                                                       
C     N       THE LENGTH OF THE ARRAY R TO BE TRANSFORMED.  THE METHOD  
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.    
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C                                                                       
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE      
C             TO BE TRANSFORMED                                         
C                                                                       
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N+15.   
C             IN THE PROGRAM THAT CALLS RFFTF. THE WSAVE ARRAY MUST BE  
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A    
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT     
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE       
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT   
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.         
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.      
C                                                                       
C                                                                       
C     OUTPUT PARAMETERS                                                 
C                                                                       
C     R       R(1) = THE SUM FROM I=1 TO I=N OF R(I)                    
C                                                                       
C             IF N IS EVEN SET L =N/2   , IF N IS ODD SET L = (N+1)/2   
C                                                                       
C               THEN FOR K = 2,...,L                                    
C                                                                       
C                  R(2*K-2) = THE SUM FROM I = 1 TO I = N OF            
C                                                                       
C                       R(I)*COS((K-1)*(I-1)*2*PI/N)                    
C                                                                       
C                  R(2*K-1) = THE SUM FROM I = 1 TO I = N OF            
C                                                                       
C                      -R(I)*SIN((K-1)*(I-1)*2*PI/N)                    
C                                                                       
C             IF N IS EVEN                                              
C                                                                       
C                  R(N) = THE SUM FROM I = 1 TO I = N OF                
C                                                                       
C                       (-1)**(I-1)*R(I)                                
C                                                                       
C      *****  NOTE                                                      
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF 
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT  
C                  SEQUENCE BY N.                                       
C                                                                       
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN      
C             CALLS OF RFFTF OR RFFTB.                                  
C                                                                       
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTB(N,R,WSAVE)                                       
C                                                                       
C     ******************************************************************
C                                                                       
C     SUBROUTINE RFFTB COMPUTES THE REAL PERODIC SEQUENCE FROM ITS      
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETER R.                                      
C                                                                       
C     INPUT PARAMETERS                                                  
C                                                                       
C     N       THE LENGTH OF THE ARRAY R TO BE TRANSFORMED.  THE METHOD  
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.    
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C                                                                       
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE      
C             TO BE TRANSFORMED                                         
C                                                                       
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N+15.   
C             IN THE PROGRAM THAT CALLS RFFTB. THE WSAVE ARRAY MUST BE  
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A    
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT     
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE       
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT   
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.         
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.      
C                                                                       
C                                                                       
C     OUTPUT PARAMETERS                                                 
C                                                                       
C     R       FOR N EVEN AND FOR I = 1,...,N                            
C                                                                       
C                  R(I) = R(1)+(-1)**(I-1)*R(N)                         
C                                                                       
C                       PLUS THE SUM FROM K=2 TO K=N/2 OF               
C                                                                       
C                        2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N)            
C                                                                       
C                       -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N)            
C                                                                       
C             FOR N ODD AND FOR I = 1,...,N                             
C                                                                       
C                  R(I) = R(1) PLUS THE SUM FROM K=2 TO K=(N+1)/2 OF    
C                                                                       
C                       2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N)             
C                                                                       
C                      -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N)             
C                                                                       
C      *****  NOTE                                                      
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF 
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT  
C                  SEQUENCE BY N.                                       
C                                                                       
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN      
C             CALLS OF RFFTB OR RFFTF.                                  
      SUBROUTINE RFFTI (N,WSAVE)                                        
      DIMENSION       WSAVE(*)                                          
CX30127 CALL SBINX (127)                                                
CY30127 CALL SBINY (127)                                                
      IF (N .EQ. 1) THEN                                                
CX40128 CALL SBOUTX (128)                                               
CY40128 CALL SBOUTY (128)                                               
      RETURN                                                            
      ENDIF                                                             
c      CALL RFFTI1 (N,WSAVE(N+1),WSAVE(2*N+1))                           
CX40128 CALL SBOUTX (128)                                               
CY40128 CALL SBOUTY (128)                                               
      RETURN                                                            
      END                                                               
