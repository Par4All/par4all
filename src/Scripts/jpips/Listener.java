/*
  $Id$
 
  $Log: Listener.java,v $
  Revision 1.5  1998/10/16 14:39:10  coelho
  BufferedReader used.

  Revision 1.4  1998/07/01 13:55:41  coelho
  fixed import.
 
  Revision 1.3  1998/07/01 13:32:25  coelho
  cleaner.
 
  Revision 1.2  1998/07/01 07:04:47  coelho
  cleaner.
 
  Revision 1.1  1998/06/30 15:00:28  didry
  Initial revision
*/

package JPips;

import java.lang.*;
import java.util.*;
import java.io.*;
import java.applet.*;

/** A class that redirects an input stream to the output stream.
  * 
  * @author Francois Didry
  */  
public class Listener 
  implements Runnable
{
  public final String listener = "listener   : ",
  public final String signal = "[tpips_wrapper] killing tpips...";

  public Resetable jpips;
  public BufferedReader	in;		//input stream from tpips
  public Pawt.PFrame frame;

  public Listener(DataInputStream in, Resetable jpips)
  {
    this.frame = frame;
    this.jpips = jpips;
    this.in = new BufferedReader(new InputStreamReader(in));
  }

  /** Listens and print the specified stream.
   */  
  public void run()
  {
    try
    {
      System.out.println(listener + "Tpips listener running");
      boolean tpipsRunning = true;
      while(tpipsRunning)
      {
	String s = in.readLine();
	System.out.println(listener + s);
	if(s.indexOf(signal) != -1) tpipsRunning = false;
      }
      System.out.println(listener+"tpips down... restarting tpips...");
      jpips.reset();
    }
    catch(IOException e)
    {
      System.out.println(e);
    }
  }
}
