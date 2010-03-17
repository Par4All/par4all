      program unstruc03

C     Check that postconditions are properly propagated forward in
C     unstructured

      read *, i

      if(i.le.5) go to 100
      j = 6
      go to 200
 100  j = 7
      i = 4
 200  continue

      print *, i, j

      end
