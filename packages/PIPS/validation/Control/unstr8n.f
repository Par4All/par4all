      program unstr8n
      integer i,j,k
      read *,j,k
      i = 4

      if(i .ge. 3) then
       i = 3
       goto 100
      else
       i = 1
       goto 100
      endif
 100  print *, j

      
      if(i .ge. 34) then
c     then goto 150
       goto 150
      else
c     else goto 150
       goto 150
      endif
 150  print *, j

      
      if(j .ge. 3) then
       i = 3
       if (k .lt. 6) then
        i = 7
        goto 200
       else
        i = i -1
        goto 200
       endif
      else
       i = 1
       goto 100
      endif
        
 200  print *, i + 6
        
      if  (k .ge. 10) then
       goto 202
      else
       goto 201
      endif
 201  if (j .ge. 11) then
       goto 202
      else
       goto 203
      endif
 202  print *, 'j >= 11'
      goto 400
 203  print *, 'j < 11'
      goto 400
      
 400  if (k .ge. 10) then
c     Some irreductible graphs
       goto 401
      else
       goto 500
      endif
 401  if (j .ge. 10) then
       goto 402
      else
       goto 403
      endif
 402  print *, '402'
      goto 403
 403  print *, '403'
      goto 402
        
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
