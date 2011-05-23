C      Check that effect on COMMON are preserved by use_def_elim

       BLOCK DATA INIT
       COMMON /PILE/ H,T
       INTEGER H,T(10)
       DATA H / 0 /
       END

       FUNCTION  DEPILE()
       COMMON  /PILE/ H,T
       INTEGER H,T(10)
         DEPILE = T(H)
         H = H - 1
       RETURN
       END

       SUBROUTINE  EMPILE(X)
       COMMON  /PILE/ H,T
       INTEGER  H,T(10),X
         H = H + 1
         T(H) = X
       RETURN
       END

       PROGRAM HELLOW
         call EMPILE( 2 )
         PRINT*, 'Depile ', DEPILE()
         STOP
       END


