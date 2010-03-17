      program unstr2

c     test of sequence restructuration by Ronan Keryell

      j = 2
      go to 100
 300  continue
      j=3
      go to 400
 100  continue
      print *, j
      go to 200
 200  continue
      go to 300
 400  continue
      print *, j

      end
