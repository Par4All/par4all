      program unstr

c     test of entry and exit node extraction by Ronan's restructurer

      j = 2

 100  continue
      print *, j
      if(j.lt.2) go to 100

      print *, j

      end
