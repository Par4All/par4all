      program unstr4

c     test of entry and exit node extraction by Ronan's restructurer
c     plus unstructured decomposition

      j = 2

 100  continue
      print *, j

      j = 2

 200  continue
      j = 3
      print *, j
      if(j.lt.3) go to 200

      if(j.lt.2) go to 100

      print *, j

      end
