/*
 * $Id$
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
