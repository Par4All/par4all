      program unstruc02

C     Check new fix-point for unstructured

      i = 0
      j = 0

 100  continue
      i = i + 1
      j = j + 2
      print *, i, j
      if(i.lt.n) go to 100

      print *, i, j

      end
