

/** $Id$
  * $Log: TextDisplayer.java,v $
  * Revision 1.1  1998/06/09 07:28:28  didry
  * Initial revision
  *
  */


package JPips;

import java.util.*;
import java.io.*;
import JPips.Pawt.*;
import java.awt.event.*;
import java.awt.*;


/** A window manager for text windows.
  * It manages the displaying of windows on the screen.
  * It regulates the amount of windows displayed at the same time.
  * @author Francois Didry
  */
public class TextDisplayer extends Displayer
{


public	Vector	frameVector;


/** Sets the maximum number of displayed windows.
  * Sets the number of displayed windows to 0.
  */
  public TextDisplayer()
    {
      super();
      frameVector = new Vector();
    }


  /** Displays a frame that contains a text file.
    */
  public void display(File file, boolean locked, boolean writable)
    {
      PTextFrame f = getFreeFrame();
      if(f != null)
        {
	  f.setTitle(file.getName());
	  f.ta.setText(getText(file));
	  f.locked = locked;
	  f.writable = writable;
	}
      else
        {
	  f = new PTextFrame(file.getName(), getText(file), locked, writable);
          register(f);
          f.setVisible(true);
	}
    }


  public void display(String title, String text,
  		      boolean locked, boolean writable)
    {
      PTextFrame f = getFreeFrame();
      if(f != null)
        {
	  f.setTitle(title);
	  f.ta.setText(text);
	  f.locked = locked;
	  f.writable = writable;
	}
      else
        {
          f = new PTextFrame(title, text, locked, writable);
          register(f);
          f.setVisible(true);
	}
    }


  public void register(PTextFrame f)
    {
      frameVector.addElement(f);
      WindowListener w = new WindowListener()
        {
	  public void windowClosing(WindowEvent e) { decNoWindows(); }
	  public void windowOpened(WindowEvent e) { incNoWindows(); }
	  public void windowClosed(WindowEvent e) {}
	  public void windowActivated(WindowEvent e) {}
	  public void windowDeactivated(WindowEvent e) {}
	  public void windowIconified(WindowEvent e) {}
	  public void windowDeiconified(WindowEvent e) {}
        };
      f.addWindowListener(w);
    }


  public PTextFrame getFreeFrame()
    {
      for(int i=0; i<frameVector.size(); i++)
        {
	  PTextFrame f = (PTextFrame)frameVector.elementAt(i);
	  if(!f.locked) return f;
	}
      return null;
    }


  public String getText(File f)
    {
      if(f.canRead())
        {
	  try
	    {
	      RandomAccessFile raf = new RandomAccessFile(f,"r");
	      String text = "";
	      String l = raf.readLine();
	      while(l != null)
	        {
	          text = text + l +"\n";
		  l = raf.readLine();
	        }
              return text;
	    }
	  catch(FileNotFoundException e)
	    {
	      System.out.println(e);
	    }
	  catch(IOException e)
	    {
	      System.out.println(e);
	    }
        }
      return null;
    }


}
