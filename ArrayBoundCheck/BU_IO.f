* Example with Input/Output command 
 
      PROGRAM IO
      PARAMETER (N=10)
      INTEGER T(N),X
     
      PRINT *, 'THIS IS A TEST'

      READ(5, 1000, END=990) T(1)
 1000 FORMAT(I5)

      stop 'ok'

 990  stop 'not enough data'

      END

