      program unstruc15

C     Triplication of a cycle: same as unstruc14 but with an unknown
C     bound for the replicated cycle

      read *, n, k
      i = 1
      if(n.gt.0) then
         j = 1
         go to 200
      endif
 100  continue
      j = 1
 200  continue
 300  continue
      j = j + 1
      if(j.lt.k) go to 300
      if(i+j.gt.105) go to 400
      i = i + 1
      if(i.lt.100) go to 100
 400  continue
      print *, i, j

      end
