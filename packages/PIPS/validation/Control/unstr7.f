      program unstr
      integer i,j,k
      read *,j,k

c     An asymetrical then test
      if(j .ge. 3) then
       i = 3
       goto 100
      endif
 100  print *, j

c     A null test
      if (k .gt. 27) goto 200
 200  print *,k

      end
