/*
 * Copyright (c) 1998, Subrahmanyam Allamaraju. All Rights Reserved.
 * 
 * Permission to use, copy, modify, and distribute this software for
 * NON-COMMERCIAL purposes and without fee is hereby granted provided that this
 * copyright notice appears in all copies.
 *
 * This software is intended for demonstration purposes only, and comes without
 * any explicit or implicit warranty.
 *
 * Send all queries about this software to sallamar@cvimail.cv.com
 *
 * Deeply extended by Ronan.Keryell@cri.ensmp.fr
 */

/*
 * $Id$
 *
 * $Log: Console.java,v $
 * Revision 1.3  1998/10/17 09:51:45  coelho
 * background color is white.
 *
 * Revision 1.2  1998/10/16 13:40:56  coelho
 * fixed.
 *
 * Revision 1.1  1998/06/30 17:35:33  coelho
 * Initial revision
 *
 * Revision 1.6  1998/05/27 06:39:33  keryell
 * Swing version.
 * Small caps package name.
 *
 * Revision 1.5  1998/04/21 12:04:47  keryell
 * Now the end of the Console is always displayed when sized up.
 *
 * Revision 1.4  1998/04/09 08:53:17  keryell
 * New version using JVC stuff to switch between a reduced size and a
 * full size ConsoleLinePanel.
 *
 * Revision 1.3  1998/04/08 14:31:48  keryell
 * New version using the CardLayout. But in fact, it does not fit the
 * requirement... :-(
 *
 * Revision 1.2  1998/04/01 13:24:45  keryell
 * Added getConsoleTextField() and getConsoleLinePanel().
 *
 * Revision 1.1  1998/03/12 16:20:45  keryell
 * Initial revision
 *
 */


package JPips;

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import com.sun.java.swing.*;
import com.sun.java.swing.border.*;


/**
 * Class Console creates a Java Console for GUI based Java Applications. Once
 * created, a Console component receives all the data directed to the standard
 * output (System.out) and error (System.err) streams. 
 * <p>
 * For example, once a Java Console is created for an application, data passed
 * on to any methods of System.out (e.g., System.out.println(" ")) and
 * System.err (e.g., stack trace in case of uncought exceptions) will be
 * received by the Console.
 * <p>
 * Note that a Java Console can not be created for Applets to run on any
 * browsers due to security violations. Browsers will not let standard output
 * and error streams be redicted (for obvious reasons).
 *
 * @author Subrahmanyam Allamaraju (sallamar@cvimail.cv.com)
 *
 * Note that if the getConsoleLinePanel() system is used, the Console
 * Frame should not be used because already display in the
 * ConsoleLinePanel.
 * @author Ronan.Keryell@cri.ensmp.fr
 */
public class Console extends JFrame implements StreamObserver
{
   JTextArea aTextArea;
   JScrollPane  the_scroll_pane;

   ObservableStream errorDevice;
   ObservableStream outputDevice;

   ByteArrayOutputStream _errorDevice;
   ByteArrayOutputStream _outputDevice;
    
   PrintStream errorStream;
   PrintStream outputStream;

   PrintStream _errorStream;
   PrintStream _outputStream;
    
   JButton clear;
   JButton close;

   JPanel console_panel, reduced_size;
   private JTextField console_text_field;
   JPanel console_line_panel;
   
   /**
    * Creates a Java Console.
    */
   public Console(String title) {
      super(title);

      /* Create a Panel to get it in the ConsoleLinePanel because the
         CardLayout freeze by using the entire Frame event if the
         Frame is a container... RK. */
      console_panel = new JPanel(new BorderLayout() {
	  public Dimension getPreferredSize() {
	      return new Dimension(300, 200);
	  }
      });
      
      addWindowListener(new WindowAdapter() {
	  public void windowClosing(WindowEvent e) 
	      {
		  setVisible(false);
	      }
      });

      // A scrollable JTextArea:
      aTextArea = new JTextArea();
      aTextArea.setEditable(false);
      aTextArea.setLineWrap(true);
      aTextArea.setBackground(Color.white);
      
      the_scroll_pane = new JScrollPane
	(aTextArea, 
	 ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS, 
	 ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);

      clear = new JButton("Clear");
      close = new JButton("Close");
	    
      clear.addActionListener(new ActionListener() {
         public void actionPerformed(ActionEvent e) {
            aTextArea.setText("");
         }
      }
                              );
	    
      close.addActionListener(new ActionListener() {
         public void actionPerformed(ActionEvent e) {
            if (console_line_panel == null)
               // The Console is an independent Frame:
               getContentPane().setVisible(false);
            else
               // The Console is inside the ConsoleLinePanel and we revert to the reduced size instead:
               setConsoleLinePanelReducedSize();
         }
      }
                              );
	    
      JPanel buttonPanel = new JPanel();
      buttonPanel.setLayout(new GridLayout(1, 0));
      buttonPanel.add(clear);
      buttonPanel.add(close);
	    
      getContentPane().setLayout(new BorderLayout());
      console_panel.add("Center", the_scroll_pane);
      console_panel.add("South", buttonPanel);
      getContentPane().add("Center", console_panel);
      
      _errorStream = System.err;
      _outputStream = System.out;

      _outputDevice = new ByteArrayOutputStream();
      _errorDevice = new ByteArrayOutputStream();
	    
	    
      this.pack();
      this.setError();
      this.setOutput();
      setEnabled(true);

   }
    
   /** A default title
    */
   public Console() {
      this("Java Console");
   }
   
   /** 
    * Clears the Console. 
    */
   public void clear() {
      try {
         outputDevice.writeTo(_outputDevice);
      }
      catch(IOException e) {
      }
      outputDevice.reset();

      try {
         errorDevice.writeTo(_errorDevice);
      }
      catch(IOException e) {
      }
      errorDevice.reset();
	   
      aTextArea.setText("");
   }


   /**
    * Sets the error device to the Console if not set already. 
    * @see #resetError
    */
   public final void setError() {
      errorDevice = new ObservableStream();
      errorDevice.addStreamObserver(this);
	    
      errorStream = new PrintStream(errorDevice, true);
	    
      System.setErr(errorStream);
   }
    
   /**
    * Resets the error device to the default. Console will no longer receive
    * data directed to the error stream.
    * @see #setError
    */
   public final void resetError() {
      System.setErr(_errorStream);
   }
    
   /**
    * Sets the output device to the Console if not set already.
    * @see #resetOutput
    */
   public final void setOutput() {
      outputDevice = new ObservableStream();
      outputDevice.addStreamObserver(this);
	    
      outputStream = new PrintStream(outputDevice, true);
	    
      System.setOut(outputStream);
   }
    
   /**
    * Resets the output device to the default. Console will no longer receive
    * data directed to the output stream.
    * @see #setOutput
    */
   public final void resetOutput() {
      System.setOut(_outputStream);
   }
    
   /**
    * Gets the minimumn size.
    */
   public Dimension getMinimumSize() {
      return new Dimension(300, 100);
   }


   /**
    * Gets the preferred size.
    */
   public Dimension getPreferredSize() {
      return getMinimumSize();
   }


   public void streamChanged() {
      aTextArea.append(outputDevice.toString());
      updateConsoleTextField(outputDevice.toString());
      try {
         outputDevice.writeTo(_outputDevice);
      }
      catch(IOException e) {
      }
      outputDevice.reset();
	    
      errorStream.checkError();
      aTextArea.append(errorDevice.toString());
      updateConsoleTextField(errorDevice.toString());
      try {
         errorDevice.writeTo(_errorDevice);
      }
      catch(IOException e) {
      }
      errorDevice.reset();
   }
    

   /**
    * Returns contents of the error device directed to it so far. Calling 
    * <a href="#clear">clear</a> has no effect on the return data of this method.
    */
   public ByteArrayOutputStream getErrorContent()
      throws IOException {
      ByteArrayOutputStream newStream = new ByteArrayOutputStream();
      _errorDevice.writeTo(newStream);
	    
      return newStream;
   }
    
   /**
    * Returns contents of the output device directed to it so far. Calling 
    * <a href="#clear">clear</a> has no effect on the return data of this method.
    */
   public ByteArrayOutputStream getOutputContent() throws IOException {
      ByteArrayOutputStream newStream = new ByteArrayOutputStream();
      _outputDevice.writeTo(newStream);
	    
      return newStream;
   }


   /** Return the previous Error PrintStream.
    */
   public PrintStream previousError() {
      return _errorStream;
   }
   

   /** Return a JextField that is updated with the last line displayed
       in the Console.
       * @author Ronan.Keryell@cri.ensmp.fr
       */
   public JTextField getConsoleTextField() {
      console_text_field = new JTextField();
      console_text_field.setEditable(false);
      console_text_field.setBackground(Color.white);
      
      return console_text_field;
   }
   

   /** Return a Panel with the last message displayed in this Console.
       With the More button you can toggle to a full-sized view of the
       Console.
       * @param title
       * @author Ronan.Keryell@cri.ensmp.fr
       */
   public JPanel getConsoleLinePanel(String title) {
      reduced_size = new JPanel(new BorderLayout())/* {
         public Dimension getPreferredSize() {
            return new Dimension(300, 10);
         }
         }*/;
      console_line_panel = new JPanel(new BorderLayout());
      console_line_panel.setBorder(BorderFactory.createTitledBorder(title));
      JButton more = new JButton("More...");

      more.addActionListener(new ActionListener() {
         public void actionPerformed(ActionEvent e) {
            // Display the full console when clicking on the button:
            console_line_panel.remove(0);
            console_line_panel.add(console_panel, BorderLayout.CENTER);
            /* Resize the window to fit the new
               console_line_panel. The side effect is that all the
               Frame is resized at the default size right now... */
            JOptionPane.getFrameForComponent(console_line_panel).pack();
            // Display the end of the Console. Well it is a side
            // effect but it works...
            aTextArea.setCaretPosition(aTextArea.getText().length());
         }
      }
                             );
      reduced_size.add(getConsoleTextField(), BorderLayout.CENTER);
      reduced_size.add(more, BorderLayout.EAST);

      // Select by default the reduced size version:
      console_line_panel.add(reduced_size, BorderLayout.CENTER);
      return console_line_panel;
   }
   
      
   /** Reduce the size of the ConsoleLinePanel.
    * @author Ronan.Keryell@cri.ensmp.fr
    */
   public void setConsoleLinePanelReducedSize() {
      // Remove the old panel...
      console_line_panel.remove(0);
      // ...and add the new one:
      console_line_panel.add(reduced_size, BorderLayout.CENTER);
      /* Resize the window to fit the new console_line_panel. The side
         effect is that all the Frame is resized at the default size
         right now... */
      JOptionPane.getFrameForComponent(console_line_panel).pack();
   }
   
   /** Update the ConsoleTextField with the end of the text displayed
    * in the Console.
    * @author Ronan.Keryell@cri.ensmp.fr
    */
   private void updateConsoleTextField(String text) {
      if (console_text_field != null) {
         // Get the last non empty line of the text:
         int line_begin;
         int line_end = text.length() - 1;
         int i;
         
         // Find the end of the significant line:
         for(i = line_end; i >= 0; i--)
            if (text.charAt(i) != '\n'
                && text.charAt(i) != '\r'
                && text.charAt(i) != ' '
                // Skip also backspace since CHESS use them
                // intensively:
                && text.charAt(i) != '\b'
                && text.charAt(i) != '\t')
               break;
         line_end = i;
         
         // Find the begin of the significant line:
         for(i = line_end - 1; i >= 0; i--)
            if (text.charAt(i) == '\n'
                || text.charAt(i) == '\b'
                || text.charAt(i) == '\r')
               break;
         line_begin = i + 1;

         if (line_begin >= 0)
            // There is still something significant:
            console_text_field.setText(text.substring(line_begin,
                                                      line_end + 1));
      }  
   }
   

   /** Launch a thread that diverts an input stream to an other one.
    * @author Ronan.Keryell@cri.ensmp.fr
    */
   void setConsoleInputStream(ObservableStream a_console_output_stream,
                              InputStream an_input_stream) {
      new ConsoleOutputStreamRun(a_console_output_stream,
                                 an_input_stream).start();
   }
   
   /** Divert both stdout and stderr to the console.
    * @author Ronan.Keryell@cri.ensmp.fr
    */
   public void setConsoleProcessOutput(Process a_process) {
      setConsoleInputStream(outputDevice, a_process.getInputStream());
      setConsoleInputStream(errorDevice, a_process.getErrorStream());
   }      
}


/** The class that waits for an input stream and writes to an
    * ObservableStream.
    * @author Ronan.Keryell@cri.ensmp.fr
    */
class ConsoleOutputStreamRun
   extends Thread {
   private ObservableStream output_stream;
   private InputStream input_stream;
   private final int BUFFER_SIZE = 400;
 
   public ConsoleOutputStreamRun(ObservableStream an_output_stream,
                                 InputStream an_input_stream) {
      output_stream = an_output_stream;
      input_stream = an_input_stream;
   }

   public void run() {      
      boolean the_end = false;
//      Debug.message("Thread \"" + this.getName() + "\" is launched.");         
   top:
      try {            
         while(true) {
            int first_offset, first_byte;
            byte buffer[] = new byte[BUFFER_SIZE];         
            int length_to_read;
            int length;
            // Just to avoid polling, wait for 1 byte first:
            first_offset = 1;            
            first_byte = input_stream.read();
            if (first_byte == -1)
               // The pipe has been closed...
               break;
            
            buffer[0] = (byte) first_byte;            
            // and then try to read the remaining stuff:
            length = input_stream.available();
            // + first_offset to output the first byte at least:
            while(length + first_offset > 0) {
               if (length > BUFFER_SIZE - first_offset)
                  length_to_read = BUFFER_SIZE - first_offset;
               else
                  length_to_read = length;
               // To be sure the output will be atomic at least for
               // the current available bytes or the output buffer,
               // concatenate the first byte with the other one:
               length_to_read =
                  input_stream.read(buffer, first_offset, length_to_read);
               if (length_to_read == -1) {
                  the_end = true;
                  // Nothing actually read:
                  length_to_read = 0;
               }
               
               length -= length_to_read;
               output_stream.write(buffer, 0,
                                   length_to_read + first_offset);
               if (the_end)
                  break top;
               
               // OK, we'll have more room in the buffer next time.
               first_offset = 0;
            }
         }
      }
      catch (Exception e) {
//         Debug.message("Thread \"" + this.getName() + "\" got exception "
//                       + e + ".");         
      }         
         
      // The pipe has been closed. No longer needed to keep this
      // thread that exits naturally...
//      Debug.message("Thread \"" + this.getName() + "\" is exiting.");         
   }     
}
