      program unstr

c     test of entry and exit node extraction by Ronan's restructurer
c     plus unstructured decomposition

 200  continue
      j = 3
      print *, j
      if(j.lt.3) go to 200

      print *, j

      end
