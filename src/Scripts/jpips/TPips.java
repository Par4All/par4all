/*
 * $Id$
 *
 * $Log: TPips.java,v $
 * Revision 1.7  1998/10/16 17:17:33  coelho
 * updates for 1.2b4
 *
 * Revision 1.6  1998/10/16 13:58:34  coelho
 * import fixed.
 *
 * Revision 1.5  1998/07/03 08:14:37  coelho
 * a little bit cleaner.
 *
 * Revision 1.4  1998/07/02 15:03:34  coelho
 * synchronized sendCommand...
 *
 * Revision 1.3  1998/07/01 13:32:34  coelho
 * more comments.
 *
 * Revision 1.2  1998/06/30 17:35:33  coelho
 * last version of FD.
 *
 * Revision 1.1  1998/06/30 16:42:54  coelho
 * Initial revision
 *
 */

package JPips;

import java.lang.*;

import java.util.*;
import java.io.*;
import java.applet.*;

import com.sun.java.swing.*;

import JPips.Pawt.*;

/** A class that creates a tpips process.
  * It defines the methods to interact with the process.  
  * 
  * @author Francois Didry
  */
public class TPips 
  implements Requestable
{
  public	Resetable	jpips;
  public	Process		tpips;		//tpips process
  public	Thread		listener,	//listener instance
  				watcher;	//watcher instance
  public	PList		list;		//contains the modules

  public	PrintWriter	out;		//output stream to tpips
  public	BufferedReader	in,		//input stream from tpips
  				inErr;		//input stream from tpips
				
  public	Vector		optionVector;	//the options of jpips
  public	TextDisplayer	textDisplayer;	//textdisplayer instance
  public	PFrame		frame;		//frame of jpips
  public	File		directory;	//current directory
  

  //the followings are tags used for the parsing of tpips messages
  public final	String		DONE = "done",
  				JPIPS = "#jpips:",
				BEGIN_REQUEST = "begin_user_request",
				BEGIN_ERROR = "begin_user_error",
				END_REQUEST = "end_user_request",
				END_ERROR = "end_user_error",
				MODULE_TAG = "$MODULES",
				TAG = "$TAG",
				INFO = "info",
				DISPLAY = "display",
				MODULES = "modules",
				SHOW = "show",
				EOL = "\n",
				SPACE = " ",
				DIR = ".",
				WORKSPACE = "workspace",
				DIRECTORY = "directory",
				NONE = "<none>",
				RESULT = "result";
  

  public final	String		execute = "tpips -jw",
  				TO_TPIPS = "to tpips   : ",
				FROM_TPIPS = "from tpips : ",
				moduleMessage = "No modules selected!",
				error = "Error";
				
				
  /** Sets the necessary instances.
    */  
  public TPips(Resetable jpips)
    {
      this.jpips = jpips;
    }


  /** Starts the tpips process and sets the streams.
    */  
  public void start()
    {
      try
        {
          System.out.println("starting tpips...");
	  tpips = Runtime.getRuntime().exec(execute);
          out = new PrintWriter(tpips.getOutputStream());
          in = new BufferedReader
	    (new InputStreamReader(tpips.getInputStream()));
	  inErr = new BufferedReader
	    (new InputStreamReader(tpips.getErrorStream()));

	  // listen to tpips error
	  Thread listener = new Thread(new Listener(inErr,jpips));
	  listener.start();

	  // warns if tpips dies.
//	  Thread watcher = new Thread(new Watcher(tpips));
//	  watcher.start();
	}
      catch(Exception e)
	{
	  System.out.println("Cannot start tpips!");
	}
    }


  /** Stops the tpips process.
    */  
  public void stop()
    {
      tpips.destroy();
    }
    

  /** Sends a quit message to tpips.
    */  
  public void quit()
    {
      out.println(replaceCommand("quit"));
      out.flush();
    }
    
    
  /** Sends an exit message to tpips.
    */  
  public void exit()
    {
      out.println(replaceCommand("exit"));
      out.flush();
    }
    
    
  /** Sends a message to tpips.
    * @param command the message
    * @return the tpips response
    */  
  synchronized public String sendCommand(String command)
    {
      String s = replaceCommand(command);
      if(s == null) return null;
      System.out.println(TO_TPIPS + s);
      out.println(s);
      out.flush();
      return getResponse();
    }
    
    
  /** Parses the messages of tpips.
    * @return the tpips response
    */  
  public String getResponse()
    {
      try
        {
	  String result = null;
    	  String s = in.readLine();
	  while(s!=null)
	    {
              System.out.println(FROM_TPIPS + s);
	      StringTokenizer t = new StringTokenizer(s,SPACE,false);
	      if(t.hasMoreTokens())
	        {
                  String response = t.nextToken();
                  if(response.equals(JPIPS))
	            {
	              response = t.nextToken();
		      if(response.equals(DONE)) 
			{
			  return result;
			}
                      else if(response.equals(DIRECTORY))
                        {
	                  result = t.nextToken(EOL).substring(1);
	                }
                      else if(response.equals(WORKSPACE))
                        {
	                  result = t.nextToken(EOL).substring(1);
			  if(result.equals(NONE)) result = null;
	                }
                      else if(response.equals(MODULES))
                        {
	                  result = t.nextToken(EOL).substring(1);
	                }
                      else if(response.equals(RESULT))
                        {
	                  result = t.nextToken(EOL).substring(1);
	                }
                      else if(response.equals(SHOW))
                        {
			  String path = t.nextToken(EOL);
			  if(path.substring(1,2).equals(DIR))
			      path = path.substring(2);
			  File f= new File(directory.getAbsolutePath()+path);
			  System.out.println(f.getAbsolutePath());
			  textDisplayer.display(f,true,true);
	                }
		      else if(response.equals(BEGIN_ERROR))
		        {
                          JOptionPane.showMessageDialog(frame,
		            getText(END_ERROR),error,
			    JOptionPane.ERROR_MESSAGE);
		        }
		      else if(response.equals(BEGIN_REQUEST))
		        {
		          String text = getText(END_REQUEST);
		        }
	            }
		}
              s = in.readLine();
	    }
	}
      catch(IOException e)
        {
	  System.out.println(e);
	}
      return null;            
    }


  /** Parses the messages of tpips until the tag is reached.
    * @param expected the expected tag for the response
    * @return the tpips response
    */  
  public String getText(String expected)
    {
      try
        {
	  String text = "";
	  while(true)
	    {
	      String s = in.readLine();
	      if(s != null && !s.equals(""))
	        {
                  if(s.equals(expected))
	            {
		      return text;
		    }
                  else
                    {
	              text = text + s;
		    }
	        }
	    }
	}
      catch(IOException e)
        {
	  System.out.println(e);
	}
      return null;            
    }


  /** Modifies a message by replacing a tag by modules names.
    * Modifies a message by replacing a marker by its true value.
    * @param message the message to modify
    * @return the new message
    */  
  public String replaceCommand(String command)
    {
      String realCommand;
      int index;
      index = command.indexOf(MODULE_TAG);
      if(index != -1)
        {
          realCommand = command.substring(0,index);
          Object tab[] = list.getSelectedValues();
          if(tab.length > 0)
	    {
	      for(int i=0; i<tab.length; i++)
                {
                  realCommand = realCommand + (String)tab[i] + SPACE;
	        }      
              command
	        = realCommand
		    + command.substring(index+MODULE_TAG.length());
	    }
	  else
	    {
              JOptionPane.showMessageDialog(frame,moduleMessage,
	      "Error",JOptionPane.ERROR_MESSAGE);
	      return null;
	    }
        }
      index = command.indexOf(TAG);
      if(index != -1)
        {
	  index = command.indexOf(SPACE);
          String marker = command.substring(TAG.length(),index);
	  realCommand = command.substring(index);
          command = replaceTAG(marker) + realCommand;
        }
      return command;
    }


  /** Parses the messages of tpips until the tag is reached.
    * @param expected the expected tag for the response
    * @return the tpips response
    */  
  public String replaceTAG(String marker)
    {
      for(int i=0; i<optionVector.size(); i++)
        {
	  Stateable op = (Stateable) optionVector.elementAt(i);
	  Vector state = op.getState();
	  if (state != null)
	    {
              for(int j=0; j<state.size(); j++)
	        {
	          PComboBox cob = (PComboBox) state.elementAt(j);
	          if(cob.marker.equals(marker))
	            {
	              int index = cob.getSelectedIndex();
		      return (String)cob.vCommand.elementAt(index);
		    }
	        }
	    }
	}
      return null;            
    }
}
