      program unstruc05

C     Check that scc's are found

      read *, i

      if(i.le.5) go to 100
      j = 6
      go to 200
 100  j = 7
 300  continue
      i = 4
 200  continue
      if(i.le.2) go to 300

      print *, i, j

      end
