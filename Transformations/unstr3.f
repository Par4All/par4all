      program unstr3

c     test of dead code elimination

      j = 2

 100  continue
      print *, j
      if(j.lt.2) go to 100

      go to 200

 300  continue
      j = 3
      print *, j

 200  continue

      end
