c     This is a test to verify that the comments 
c     are kept through unspaghettify 
      program unstr
      integer i,j,k
      read *,j,k
      i = 4

      if(i .ge. 34) then
c     then goto 150
       goto 150
      else
c     else goto 150
       goto 150
      endif
 150  print *, j

c     goto 200
      goto 200
c     goto 300
 200  goto 300
c     stop
 300  stop
      end
