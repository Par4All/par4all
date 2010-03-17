      program unstruc04

C     Check that postconditions are properly propagated forward in
C     unstructured with unreachable code and multiple joint

      read *, i

      if(i.le.5) go to 100
      j = 6
      go to 200
 300  continue
      i = 5
 100  j = 7
      if(i.le.2) go to 200
      i = 4
 200  continue

      print *, i, j

      end
