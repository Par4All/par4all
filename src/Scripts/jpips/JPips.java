/*
 * $Id$
 *
 * $Log: JPips.java,v $
 * Revision 1.5  1998/10/17 12:21:09  coelho
 * indentation fixed.
 * colors fixed.
 *
 * Revision 1.4  1998/10/16 17:17:18  coelho
 * updates for 1.2b4
 *
 * Revision 1.3  1998/07/03 11:55:17  coelho
 * workspace menu moved to workspace.
 *
 * Revision 1.2  1998/07/01 15:55:22  coelho
 * jpips.menus property used for the menus file.
 *
 * Revision 1.1  1998/06/30 17:34:07  coelho
 * Initial revision
 *
 */

package JPips;

import java.lang.*;
import java.util.*;
import java.io.*;

import java.awt.event.*;
import java.applet.*;
import java.awt.*;

//import java.awt.swing.preview.*;
import com.sun.java.swing.*;
import com.sun.java.swing.event.*;
import com.sun.java.swing.text.*;
import com.sun.java.swing.border.*;

import JPips.Pawt.*;

/** The application of jpips.
  * It builds the main frame and its menu.
  * 
  * @author Francois Didry
  */
public class JPips 
  extends Applet 
  implements Resetable
{


  static public final String	
      source = "jpips.menus",	//the parsed text for the menu
      TOP_MENU = "TOP_MENU";	//tag to delimit a menu

  public DirectoryManager	directoryManager;	//manages the directory
  public WorkspaceManager	workspaceManager;	//manages the workspace
  public ModuleManager		moduleManager;		//manages the modules
  
  public TPips		tpips;		// tpips instance
  public TextDisplayer 	textDisplayer;	// regulates the displayed windows
  public Vector		optionVector;	// contains the components for JPips
			
  public PMenuBar	menu;		// menu of JPips

  static public boolean	console;	// notifies the use of a console
  static public	PFrame	frame;		// main frame

  public PPanel		filePanel;      // main panel...
  public PMenuBar 	mb;

  static public final String title = "JPips";
  static public final String noConsole = "-n";

  public JPips() 
  { 
    super(); 
  }

  /** Creates the main frame with JPips inside.
    */
  static public void main(String args[]) 
  {
    console = true;
    if(args.length>0 && args[0].equals(noConsole)) console = false;
    frame = new PFrame(title);
    GridBagLayout fLayout = new GridBagLayout();
    frame.getContentPane().setLayout(fLayout);
    JPips jpips = new JPips();
    jpips.init();
  }

  /** Launches TPips.
    * Builds the JPips main frame.
    */
  public void init() 
  {
    super.init();
    
    tpips = new TPips(this);
    tpips.start();
    
    GridBagConstraints c = new GridBagConstraints();
    optionVector = new Vector();
    
    buildJPipsPanel();
    buildOptionVector();
    frame.setJMenuBar(getMenuBar());
    add((Container)frame.getContentPane(),getJPipsPanel(),
	0,0,1,1,1,1,1.0,1.0,5,
	GridBagConstraints.BOTH,GridBagConstraints.WEST,c);
    
    frame.lock(true);
    frame.pack();
    frame.setVisible(true);
  }
    
    
  /** Resets jpips.
    */
  public void reset() 
  {
    tpips.start();
    directoryManager.reset();
    workspaceManager.reset();
    moduleManager.reset();
    frame.lock(true);
  }
    
    
  /** Builds the vector of Option objects.
    */
  public void buildOptionVector()
  {
    optionVector.addElement(workspaceManager.getOption());
    
    try
    {
      FileReader f = new FileReader(System.getProperty(source));
      Parser p = new Parser(f);
      String lineContent = p.nextNonEmptyLine();
      while(lineContent != null)
      {
	if(lineContent.equals(TOP_MENU))
	{
	  OptionParser op = new OptionParser(p, tpips);
	  optionVector.addElement
	    (new Option(op.title,op.menu,op.frame,op.vector,op.state));
	}
	lineContent = p.nextNonEmptyLine();
      } 
    }
    catch(FileNotFoundException e)
    {
      System.out.println(e);
    }
    
    optionVector.addElement(getHelpOption());
    frame.optionVector = optionVector;
    tpips.optionVector = optionVector;
  }


  /** @return the menu of the main frame
    */
  public PMenuBar getMenuBar()
  {
    PMenuBar mb = new PMenuBar();
    for(int i=0; i<optionVector.size(); i++)
      mb.add(((Option)optionVector.elementAt(i)).getMenu());
    return mb;
  }


  /** @return the panel of the main frame
    */
  public PPanel getJPipsPanel()
  {
    return filePanel;
  }
  
  public void buildJPipsPanel()
  {
    filePanel = new PPanel(new GridBagLayout());
    GridBagConstraints c = new GridBagConstraints();
    PPanel p;
    PButton b;
    ActionListener a;
    PTextField tf;
    
      //displayer panel
    textDisplayer = new TextDisplayer(frame);
    add((Container)filePanel,(PPanel)textDisplayer.getComponent(),
	1,2,2,1,1,1,1.0,0.0,5,
	GridBagConstraints.NONE,GridBagConstraints.EAST,c);
    tpips.textDisplayer = textDisplayer;
    
    //directory
    directoryManager = new DirectoryManager(frame, tpips);
    p = (PPanel)directoryManager.getComponent();
    add((Container)filePanel,p,0,0,3,1,1,1,1.0,0.0,5,
	GridBagConstraints.HORIZONTAL,GridBagConstraints.WEST,c);
    
    //modules
    moduleManager = new ModuleManager(tpips);
    p = (PPanel)moduleManager.getComponent();
    add((Container)filePanel,p,0,2,1,1,1,1,0.0,0.0,5,
	GridBagConstraints.NONE,GridBagConstraints.WEST,c);
    tpips.list = moduleManager.list;
    
    //workspace
    workspaceManager = new WorkspaceManager(tpips, frame, directoryManager,
					    moduleManager, textDisplayer);
    p = (PPanel)workspaceManager.getComponent();
    add((Container)filePanel,p,0,1,1,1,1,1,0.0,0.0,5,
	GridBagConstraints.NONE,GridBagConstraints.WEST,c);
    
    //console
    if(console)
    {
      Console cons = new Console();
      add((Container)filePanel,cons.getConsoleLinePanel
	  ("TPips console"),0,3,3,1,1,1,1.0,0.0,5,
          GridBagConstraints.BOTH,GridBagConstraints.WEST,c);
    }
  }
  
  /** @return the option object for the help menu
    */
  public Option getHelpOption()
  {
    PMenuItem mi;
    
    PMenu m = new PMenu("Help");
    mi = (PMenuItem) m.add(new PMenuItem("About"));
    mi = (PMenuItem) m.add(new PMenuItem("Introduction"));
    mi = (PMenuItem) m.add(new PMenuItem("Documentation"));
    
    return new Option(m.getName(), m, null, null, null);
  }

  /** A short add method for a GridBagLayout.
    */
  public void add(Container cont, Component comp, int x, int y, int w, int h,
  		  int px, int py, double wex, double wey, int in,
		  int f, int a, GridBagConstraints c)
  {
    c.insets = new Insets(in,in,in,in);
    c.gridx = x;
    c.gridy = y;
    c.gridwidth = w;
    c.gridheight = h;
    c.ipadx = px;
    c.ipady = py;
    c.weightx = wex;
    c.weightx = wex;
    c.fill = f;
    c.anchor = a;
    cont.add(comp,c);
  }
}
