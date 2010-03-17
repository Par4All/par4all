      program unstruc14

C     Triplication of a cycle: it does not work because Bourdoncle's
C     procedure happens to break both cycles with node "j=j+1"

      read *, n
      i = 1
      if(n.gt.0) then
         j = 2
         go to 200
      endif
 100  continue
      j = 1
 200  continue
 300  continue
      j = j + 1
      if(j.lt.10) go to 300
      if(i+j.gt.105) go to 400
      i = i + 1
      if(i.lt.100) go to 100
 400  continue
      print *, i, j

      end
