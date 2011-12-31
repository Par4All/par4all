      program asgoto

C     Check desugaring of assigned go to's and elimination of carriage returns

      assign 100 to i

      assign 200 to i

      go to 600

      go to i, (100, 300) 

 600  go to i, (400)

 500  go to i, (100, 300, 400)

 100  continue
      print *, 'I reach 100'
      print *, i

      stop
 300  continue
      print *, 'I reach 300'
      stop
 200  continue
      print *, 'I reach 200'
      stop
 400  continue
      print *, 'I reach 400'
      go to 500
      end
