      program unstr
      integer i,j,k
      read *,j,k
      i = 4

c     500      
 500  if (j .ge. 11) then
c     goto 501
       goto 501
      else
c     goto 502
       goto 502
      endif
c     501
 501  print *, '501'
c     goto 502      
      goto 502
 502  print *, '502'
      goto 501

      end
