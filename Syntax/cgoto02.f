      program cgoto02

      read *, i,j

      goto(100,200,300,400), i+j
      j = 1
 100  continue
      j = 2
 200  continue
      j = 3
 300  continue
      j = 4
 400  continue
      j = 5
      end
