      PROGRAM SPLIT_PRAGMA2

      INTEGER NX,NY,NZ,AX,ISH,JSH,KSH,I,J,K,IP1,JP1,KP1
      REAL*8 KT,KX,KY,KZ,GM,RE,PR,AL0,AL1,AL2,AL3,AL4,AL5,AL6,STEP

      REAL*8 Q(5,10,10,10)

      REAL*8 JE(5,5,10,10,10),JV(5,5,10,10,10)

      REAL*8 U,V,W,FI2,ALF,THT,A1,MU,RO,US,VS,WS,ROS

      DO I = 1, NX
         IP1 = MOD(I, NX+1-ISH)+ISH
         ROS = Q(1,IP1,JP1,KP1)
         US = Q(2,IP1,JP1,KP1)/ROS
         VS = Q(3,IP1,JP1,KP1)/ROS
         WS = Q(4,IP1,JP1,KP1)/ROS
         RO = Q(1,I,J,K)
         U = Q(2,I,J,K)/RO
         V = Q(3,I,J,K)/RO
         JE(2,3,I,J,K) = KY*U-(GM-1.0D0)*KX*V
         W = Q(4,I,J,K)/RO
         FI2 = 0.5D0*(GM-1.0D0)*(U*U+V*V+W*W)
         THT = KX*U+KY*V+KZ*W
         MU = ((GM-1.0D0)*(Q(5,I,J,K)/RO-0.5D0*(U*U+V*V+W*W)))
     &        **0.75D0
         JV(2,1,I,J,K) = MU/STEP*(AL1*(U/RO-US/ROS)+AL2*(V/RO-
     &        VS/ROS)+AL3*(W/RO-WS/ROS))
         ALF = GM*Q(5,I,J,K)/RO
         A1 = ALF-FI2
      ENDDO
      END
