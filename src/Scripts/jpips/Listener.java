/*
 $Id$
 */

package fr.ensmp.cri.jpips;

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
  public final String listener = "listener   : ";
  public final String signal = "[tpips_wrapper] killing tpips...";
  
  public Resetable jpips;
  public BufferedReader in;  //input stream from tpips
  public Pawt.PFrame frame;
  
  public Listener(BufferedReader in, Resetable jpips)
  {
    this.frame = frame;
    this.jpips = jpips;
    this.in = in;
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
