/*
 * $Id$
 *
 * $Log: Listener.java,v $
 * Revision 1.2  1998/07/01 07:04:47  coelho
 * cleaner.
 *
 * Revision 1.1  1998/06/30 15:00:28  didry
 * Initial revision
 */

package JPips;

import java.lang.*;
import java.util.*;
import java.io.*;
import java.applet.*;

import JPips.Pawt.*;

/** A class that redirects the tpips output stream.
  * 
  * @author Francois Didry
  */  
public class Listener 
  implements Runnable
{
  public	Resetable	jpips;
  public	DataInputStream	in;		//input stream from tpips
  public final	String 		listener = "listener   : ",
  				signal = "[tpips_wrapper] killing tpips...";

  public PFrame frame;

  public Listener(DataInputStream in, Resetable jpips)
    {
      this.frame = frame;
      this.in = in;
      this.jpips = jpips;
    }

  /** Listens and print the specified stream.
    */  
  public void run()
    {
      try
        {
	  System.out.println(listener+"Tpips listener running");
	  boolean tpipsRunning = true;
          while(tpipsRunning)
	    {
	      String s = in.readLine();
	      System.out.println(listener + s);
	      if(s.indexOf(signal) != -1) tpipsRunning = false;
	    }
	  System.out.println(listener+"tpips down...restarting tpips...");
	  jpips.reset();
	}
      catch(IOException e)
        {
	  System.out.println(e);
	}
    }
}
