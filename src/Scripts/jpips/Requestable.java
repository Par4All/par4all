/*
 * $Id$
 *
 * $Log: Requestable.java,v $
 * Revision 1.1  1998/06/30 17:34:50  coelho
 * Initial revision
 *
 */

package JPips;

/** An interface that defines interactions between jpips and a process.
  * 
  * @author Francois Didry
  */
public interface Requestable
{
  public void start();
  public void stop();
  public String sendCommand(String command);
}
