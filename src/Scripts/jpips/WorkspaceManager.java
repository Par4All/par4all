
/** $Id$
  * $Log: WorkspaceManager.java,v $
  * Revision 1.2  1998/07/03 11:55:30  coelho
  * workspace menu moved here.
  *
  * Revision 1.1  1998/06/30 17:35:33  coelho
  * Initial revision
  *
  */


package JPips;


import java.lang.*;
import java.util.*;
import java.io.*;
import java.awt.*;
import java.awt.swing.*;
import JPips.Pawt.*;
import java.awt.event.*;
import java.awt.swing.preview.*;
import java.awt.swing.border.*;
import java.lang.*;
import java.awt.swing.event.*;
import java.awt.swing.text.*;


/** A workspace file for JPips
  * @author Francois Didry
  */  
public class WorkspaceManager implements JPipsComponent
{


  public TPips			tpips;		// tpips instance
  public PFrame			frame;		// frame of jpips
  public DirectoryManager	directoryManager;	// manager instance
  public ModuleManager		moduleManager;		// manager instance
  public TextDisplayer		textDisplayer;		// displayer instance
  public PTextField		tf,		// textfield of the panel
				editText;	// gets a new workspace name
  
  public File			workspace;	// current workspace
  public PDialog		dialog;		// opened to create a workspace
  public PPanel			panel;		// jpips workspace panel
  public PList			filesList;	// contains fortran files
  public PMenu 			wk_menu;	// menu.
  private Vector 		v1, v2; 	// used by Option...
  private Option		option;


  /** Creates a workspace file.
    * Creates the workspace component.
    */  
  public WorkspaceManager(TPips tpips, 
			  PFrame frame,
			  DirectoryManager directoryManager, 
			  ModuleManager moduleManager,
			  TextDisplayer textDisplayer)
    {
      this.tpips = tpips;
      this.frame = frame;
      this.directoryManager = directoryManager;
      this.moduleManager = moduleManager;
      this.textDisplayer = textDisplayer;
      buildPanel();
      buildMenu();
    }



  /** Creates the workspace panel for jpips.
    */  
  public void buildPanel()
    {
      panel = new PPanel(new BorderLayout());
      panel.setBorder(new TitledBorder("Current workspace"));
      tf = new PTextField();
      tf.setPreferredSize(new Dimension(200,20));
      tf.setEnabled(false);
      panel.add(tf,BorderLayout.WEST);
    }


  /** @return the workspace panel for JPips
    */  
  public Component getComponent()
    {
      return (Component)panel;
    }



  /** Checks if the tpips and the jpips workspace are similar.
    */  
  public boolean check()
    {
      String s = tpips.sendCommand("info workspace");
      if(s==null) return false;
      workspace = new File(s);
      tf.setText(workspace.getName());
      return true;
    }


  /** Opens a workspace in tpips.
    */  
  public void open(File f)
    {
      workspace = f;
      directoryManager.open(new File(f.getParent()));
      tpips.sendCommand("open "+workspace.getName());
      if(check())
        {
          option.setState(false);
          moduleManager.setModules();
          directoryManager.setActivated(false);
          frame.lock(false);
	}
      else
        {
	  workspace = null;
	}
    }


  /** Launches a checkpoint in tpips.
    */  
  public void checkPoint()
    {
      tpips.sendCommand("checkpoint");
    }


  /** Closes a workspace in tpips.
    */  
  public void close(boolean delete)
    {
      tf.setText("");
      tpips.sendCommand("close ");
      option.setState(true);
      frame.lock(true);
      directoryManager.setActivated(true);
      textDisplayer.closeAll();
      moduleManager.unsetModules();
      if(delete) delete();
      else workspace = null;
    }
    

  /** Deletes a workspace in tpips.
    */  
  public void delete()
    {
      String s = workspace.getAbsolutePath();
      int index = s.lastIndexOf("/");
      s = s.substring(index+1);
      tpips.sendCommand("delete "+s);
      workspace = null;
    }
    
    
  /** Displays a dialog frame and a file chooser to get a name and modules.
    * Creates a workspace in tpips.
    */
  public void create()
    {
      dialog = new PDialog(frame, "New Workspace",true);
      dialog.getContentPane().setLayout(new GridBagLayout());
      GridBagConstraints c = new GridBagConstraints();
      PButton b;
      PPanel p;
      ActionListener a;
      
      //name
      p = new PPanel(new BorderLayout());
      p.setBorder(new TitledBorder("Name"));
      editText = new PTextField();
      editText.setPreferredSize(new Dimension(150,20));
      p.add(editText, BorderLayout.WEST);
      add((Container)dialog.getContentPane(),p,0,0,1,1,1,1,0.0,0.0,5,
          GridBagConstraints.NONE,GridBagConstraints.WEST,c);
      
      //list
      p = new PPanel(new GridBagLayout());
      p.setBorder(new TitledBorder("Attached files"));
      GridBagConstraints cc = new GridBagConstraints();
      filesList = new PList(new DefaultListModel());
      filesList.setSelectionMode(2);
      PScrollPanel s = new PScrollPanel((Component)filesList);
      s.setPreferredSize(new Dimension(400,100));
      MouseListener m = new MouseAdapter()
        {
          public void mouseClicked(MouseEvent e)
	    {
              if (e.getClickCount() == 2)
	        {
	          DefaultListModel dlm
		    = (DefaultListModel) filesList.getModel();
		  dlm.removeElement(filesList.getSelectedValue());
                }
            }
        };
      filesList.addMouseListener(m);
      add((Container)p,s,0,0,2,1,1,1,1.0,1.0,0,
          GridBagConstraints.BOTH,GridBagConstraints.WEST,cc);
      
      //add & remove
      PPanel p2 = new PPanel(new GridLayout(1,0));
      b = new PButton("Add...");
      a = new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    {
	      JFileChooser chooser = new JFileChooser(
	        directoryManager.getDirectory());
	      String st[] = new String[2]; st[0] = ".f"; st[1] = ".F";
	      FileType types[] =  new FileType[1];
	      types[0] = new FileType.ExtensionBased
		           ("Fortran files (*.f,*.F)", st, null);
	      chooser.setChoosableFileTypes(types);
	      if(chooser.showDialog(frame) == 0)
                {
	          Vector v = chooser.getDirectoryPane().getSelectedFiles();
	          if(v != null)
		    {
	              DefaultListModel dlm = 
			(DefaultListModel) filesList.getModel();
		      for(int i=0; i<v.size(); i++)
	                {
		          boolean ok = true;
			  File f = (File)v.elementAt(i);
			  if(!f.isDirectory()
			     &&( (f.getName().indexOf(".f")!=-1)
			      || (f.getName().indexOf(".F")!=-1)))
			    {
	                      for(int j=0; j<dlm.size(); j++)
			        {
			          if(((String)dlm.elementAt(j))
				    .equals(f.getAbsolutePath()))ok = false;
				}
			      if(ok) dlm.addElement(f.getAbsolutePath());
			    }
			  else
			    {
                              JOptionPane.showMessageDialog(frame,
		                "Invalid Fortran file!","Error",
			        JOptionPane.ERROR_MESSAGE);
			    }
		        }
		    }
	        }
	      dialog.pack();
	    }
	};
      b.addActionListener(a);
      p2.add(b);
      b = new PButton("Remove");
      a = new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    {
	      DefaultListModel dlm = (DefaultListModel) filesList.getModel();
	      if(dlm.size() > 0)
	        {
		  Object o[] = filesList.getSelectedValues();
	          for(int i=0; i<o.length; i++) dlm.removeElement(o[i]);
		}
	      dialog.pack();
	    }
	};
      b.addActionListener(a);
      p2.add(b);
      add(p,p2,0,1,2,1,1,1,1.0,1.0,5,
          GridBagConstraints.BOTH,GridBagConstraints.WEST,c);
      add((Container)dialog.getContentPane(),p,0,1,2,1,1,1,1.0,1.0,5,
          GridBagConstraints.BOTH,GridBagConstraints.WEST,c);

      //ok & cancel
      p = new PPanel(new GridLayout(1,0));
      b = new PButton("Ok");
      a = new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    {
              DefaultListModel dlm
	        = (DefaultListModel) filesList.getModel();	      
	      if(!editText.getText().equals(""))
	        {
		  if(dlm.size() != 0)
	            {
		      File f = new File(
		        directoryManager.getDirectory()
		        +"/"+editText.getText());
		      File truefile
		        = new File(f.getAbsolutePath()+".database");
		      if(!truefile.exists())
		        {		  
		          dialog.setVisible(false);
                          createWorkspace(f,filesList);
		          frame.pack();
			}
		      else
		        {
                          JOptionPane.showMessageDialog(frame,
		            "File "+editText.getText()+" exists!","Error",
			    JOptionPane.ERROR_MESSAGE);
			}
		    }
		  else
		    {
                      JOptionPane.showMessageDialog(frame,
		        "Add Fortan files!","Error",JOptionPane.ERROR_MESSAGE);
		    }
		}
	      else
	        {
                  JOptionPane.showMessageDialog(frame,
		    "Enter a name!","Error",JOptionPane.ERROR_MESSAGE);
		}
	    }
	};
      b.addActionListener(a);
      p.add(b);
      b = new PButton("Cancel");
      a = new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    {
	      dialog.setVisible(false); 
	      frame.pack();
	    }
	};
      b.addActionListener(a);
      p.add(b);
      add((Container)dialog.getContentPane(),p,0,2,1,1,1,1,1.0,0.0,5,
          GridBagConstraints.HORIZONTAL,GridBagConstraints.WEST,c);

      dialog.pack();
      dialog.setLocationRelativeTo(frame);
      dialog.setVisible(true);
    }


  /** Creates a workspace in tpips.
    */  
  public void createWorkspace(File f, PList l)
    {
      directoryManager.check();
      workspace = f;
      tf.setText(workspace.getName());
      DefaultListModel dlm = (DefaultListModel) l.getModel();
      String command = "create "+ workspace.getName();;
      for(int i=0; i<dlm.size(); i++)
	{
	  String pathFile = (String)dlm.elementAt(i);
	  command = command + " " + pathFile;
        }
      tpips.sendCommand(command);
      option.setState(false);
      moduleManager.setModules();
      check();
      directoryManager.setActivated(false);
      frame.lock(false);
    }


  /** Displays a file chooser.
    * Opens a workspace in tpips.
    */
  public void choose()
    {
      JFileChooser chooser = new JFileChooser(
        directoryManager.getDirectory());
      String st[] = new String[1]; st[0] = "database";
      FileType types[] =  new FileType[1];
      types[0] = new FileType.ExtensionBased
		       ("Workspace (*.database)", st, null);
      chooser.setChoosableFileTypes(types);
      if(chooser.showDialog(frame) == 0)
	{
	  File f = chooser.getDirectoryPane().getSelectedFile();
	  if(f == null) f = chooser.getDirectoryPane().getCurrentDirectory();
	  String s = f.getAbsolutePath();
          int index = s.indexOf(".");
          if(index != -1 && s.substring(index+1).equals("database"))
            {
	      s = s.substring(0,index);
              open(new File(s));
            }
          else
            {
              JOptionPane.showMessageDialog(frame,
		"Invalid workspace!","Error",JOptionPane.ERROR_MESSAGE);
	    }
	}
      frame.pack();
    }


  /** A short add method for a GridBagLayout.
    */
  public void add(Container cont, Component comp, int x, int y, int w, int h,
  		  int px, int py, double wex, double wey, int in,
		  int f, int a, GridBagConstraints c)
    {
      c.insets = new Insets(in,in,in,in);
      c.gridx = x;
      c.gridy = y ;
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


  public PMenu getMenu()
    {
      return wk_menu;
    }
  
  /** @return the option object for the workspace menu
    */
  public Option getOption()
    {
      return option;
    }

  /** Build the workspace menu.
    */
  public void buildMenu()
    {
      PMenuItem mi;
      PCheckBoxMenuItem cbmi;

      wk_menu = new PMenu("Workspace");
      v1 = new Vector();
      v2 = new Vector();

      //create
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Create"));
      mi.addActionListener(new ActionListener()
	{
	  public void actionPerformed(ActionEvent e)
	    { create(); }
	});
      v1.addElement(mi);

      //open
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Open"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { choose(); }
	});
      v1.addElement(mi);

      //checkpoint
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Checkpoint"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { checkPoint(); }
	});
      mi.setEnabled(false);
      v2.addElement(mi);

      //close
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Close"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { close(false); }
	});
      mi.setEnabled(false);
      v2.addElement(mi);
      
      //quit
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Quit"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { tpips.quit(); System.exit(0); }
	});
      v1.addElement(mi);

      //close & quit
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Close & Quit"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { close(false); tpips.quit(); System.exit(0);}
	});
      mi.setEnabled(false);
      v2.addElement(mi);
      
      //close & delete & quit
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Delete & Quit"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    {
	      close(true);
	      tpips.quit();
	      System.exit(0);
	    }
	});
      mi.setEnabled(false);
      v2.addElement(mi);
      
      //exit
      mi = (PMenuItem) wk_menu.add(new PMenuItem("Exit"));
      mi.addActionListener(new ActionListener()
        {
	  public void actionPerformed(ActionEvent e)
	    { tpips.exit(); System.exit(0); }
	});
      mi.setForeground(Color.red);

      option = new Option(wk_menu.getName(), wk_menu, null, null, null, v1,v2);
    }

  public void setActivated(boolean yes) {}

  public void reset()
    {
      tf.setText("");
      option.setState(true);
      workspace = null;
    }


}


