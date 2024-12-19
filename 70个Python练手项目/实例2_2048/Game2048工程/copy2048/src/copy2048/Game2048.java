/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package copy2048;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.swing.JButton;

/**
 *
 * @author Jack
 */
public class Game2048 extends javax.swing.JFrame {
    private JButton[][] buttonsNames = new JButton[4][4];//这个呢，是用来模拟2048的
    private List<JButton> list = new ArrayList<JButton>();
    Random random = new Random();//创建随机数对象
    private int times = 16;//这个呢，是用来记录随机的方块数
    private int scores  = 0;//这个呢是用来记录分数的
    private int l1,l2,l3,l4;//来标记四个方向的移动
    /**
     * Creates new form Game2048
     */
    public Game2048() {
        initComponents();
        //初始化
        initialise();
        ScoreField.addKeyListener(new KeyAdapter() {
            public void keyPressed(KeyEvent e){
                KeyMove(e);
            }

            private void KeyMove(KeyEvent e) {
                //向上的移动
                int Code = e.getKeyCode();//来获取当前所按下了什么键
                String str=null;//我们把他定义为当前的数字
                String str1=null;//我们把他定义为当前的前面的数字
                int num;//来记录分数的一伴
                switch(Code){
                 //进行向上的移动
                    case KeyEvent.VK_UP://代表上键
                    case KeyEvent.VK_W:
                        for(int y = 1;y<4;y++ ){
                            //第一个for用来便利当前的行的数目
                            for(int x=0;x<4;x++){
                                //这个呢是用来便利当前的列的数目
                                
                                   str = buttonsNames[x][y].getText();//获取当前的数字
                                   if(str.compareTo("0")==0){
                                       //判断当前的元素是否为空
                                       continue;//为空退出本次循环
                                   }
                                   int dispos = -1;//记录本次循环的位置
                                for(int y1=y-1;y1>=0;y1--){
                                    //这个循环是用来便利当前元素的上面的所有元素
                                    str1 = buttonsNames[x][y1].getText();//来获取下一个元素的值
                                    if(str1.compareTo("0")!=0){
                                        //判断当前元素是否为空
                                        //不为空继续下次操作
                                        //判断一下是否便利的当前的下元素和当前的x,y的元素是否相同
                                        if(str.compareTo(str1)==0){
                                            //判断是否相同，如果相同就合并，就不需要移动了
                                            dispos = -1;//表示不进行了移动
                                            
                                            buttonsNames[x][y].setText("0");
                                            num = Integer.parseInt(str);//把记一次记录的当前数的值，转化为数字
                                            scores = num*2;
                                            
                                            ScoreField.setText(scores+"");//赋值分数文本框
                                            //由于当前合并了，所以times++，代表空元素多了一个
                                            times++;
                                            
                                            str = String.valueOf(2*num);
                                            
                                            buttonsNames[x][y1].setText(str);
                                        }
                                        //如果当前的元素和便利的他的上一个元素不相同的话，就结束循环
                                        break;
                                        
                                    }
                                    else{
                                        //如果为空就代表可以移动,所以就记录位置
                                        dispos = y1;
                                    }
                                }
                                if(dispos>-1){
                                    //不等与-1就代表可以移动
                                    buttonsNames[x][y].setText("0");
                                    buttonsNames[x][dispos].setText(str);//实现移动的效果，就是通过当前位置的值赋值给找到的上面的不为空的元素位置
                                }
                            }
                        }
                        l1 = 1;//代表确定当前已经移动了
                        Create2();
                        break;
                }
                
            }
        } );
        //一开始我们要创建两次
        Create2();
        Create2();
    }
     private void Create2() {
         boolean r = false;
         int i,j;//用来表示随机数行列标识
         String str = null;
         if(times>0){
             while(!r){
                 i = random.nextInt(4);//得到一个0-3之间的随机数
                 j = random.nextInt(4);//得到一个0-3之间的随机数
                 str = buttonsNames[i][j].getText();//用来获取随机生成的文本值
                 if(str.compareTo("0")==0){
                     //如何为0的话就返回0
                     buttonsNames[i][j].setText("2");
                     times--;//代表当前成功创建了一个了，所以减少了一个空白的个数
                     r=true;
                     l1 = l2 =l3 = l4 = 0;
                 }
             }
         }
         else if(l1>0&&l2>0&&l3>0&&l4>0){
             tips.setText("Game Over");//如过四个方向的标识都被设置为1的时候代表不能移动了，代表游戏结束了，
         }
     }
  private void initialise() {
        //初始化
      list.add(Button1);
      list.add(Button2);
      list.add(Button3);
      list.add(Button4);
      list.add(Button5);
      list.add(Button6);
      list.add(Button7);
      list.add(Button8);
      list.add(Button9);
      list.add(Button10);
      list.add(Button11);
      list.add(Button12);
      list.add(Button13);
      list.add(Button14);
      list.add(Button15);
      list.add(Button16);
      
      //咱们呢,接下来就通过集合来给二维数组赋值
      for(int i = 0;i<list.size();i++){
          buttonsNames[i%4][i/4] = list.get(i);
      }
    }
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        Button16 = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        Button1 = new javax.swing.JButton();
        Button4 = new javax.swing.JButton();
        Button5 = new javax.swing.JButton();
        Button9 = new javax.swing.JButton();
        Button6 = new javax.swing.JButton();
        Button10 = new javax.swing.JButton();
        Button8 = new javax.swing.JButton();
        Button11 = new javax.swing.JButton();
        Button2 = new javax.swing.JButton();
        Button12 = new javax.swing.JButton();
        Button7 = new javax.swing.JButton();
        Button13 = new javax.swing.JButton();
        tips = new javax.swing.JLabel();
        Button14 = new javax.swing.JButton();
        Button3 = new javax.swing.JButton();
        Button15 = new javax.swing.JButton();
        ScoreField = new javax.swing.JTextField();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        Button16.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button16.setText("0");

        jLabel1.setFont(new java.awt.Font("幼圆", 0, 36)); // NOI18N
        jLabel1.setText("   分数：");

        Button1.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button1.setText("0");
        Button1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                Button1ActionPerformed(evt);
            }
        });

        Button4.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button4.setText("0");

        Button5.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button5.setText("0");

        Button9.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button9.setText("0");

        Button6.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button6.setText("0");

        Button10.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button10.setText("0");

        Button8.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button8.setText("0");

        Button11.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button11.setText("0");

        Button2.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button2.setText("0");
        Button2.setMaximumSize(new java.awt.Dimension(45, 45));
        Button2.setMinimumSize(new java.awt.Dimension(45, 45));
        Button2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                Button2ActionPerformed(evt);
            }
        });

        Button12.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button12.setText("0");

        Button7.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button7.setText("0");

        Button13.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button13.setText("0");

        tips.setFont(new java.awt.Font("方正兰亭超细黑简体", 1, 18)); // NOI18N
        tips.setText("Tips：使用上、下、左、右键或者W、S、A、D键控制");

        Button14.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button14.setText("0");

        Button3.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button3.setText("0");

        Button15.setFont(new java.awt.Font("Tempus Sans ITC", 1, 36)); // NOI18N
        Button15.setText("0");

        ScoreField.setFont(new java.awt.Font("Segoe Script", 1, 36)); // NOI18N
        ScoreField.setText("0");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(31, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(Button13, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button9, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button5, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addGap(18, 18, 18)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(Button2, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button6, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button10, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button14, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)))
                            .addComponent(jLabel1, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 178, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(21, 21, 21)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(Button15, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button11, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button7, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button3, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addGap(14, 14, 14)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(Button8, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button4, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button12, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(Button16, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(43, 43, 43)
                                .addComponent(ScoreField, javax.swing.GroupLayout.PREFERRED_SIZE, 193, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(28, 28, 28))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addComponent(tips, javax.swing.GroupLayout.PREFERRED_SIZE, 490, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(36, 36, 36))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(43, 43, 43)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(ScoreField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 73, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(Button2, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(Button4, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addComponent(Button1, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addComponent(Button3, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(Button5, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button6, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button7, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button8, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(Button12, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button11, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button10, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button9, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(Button13, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button14, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button15, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(Button16, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addComponent(tips, javax.swing.GroupLayout.PREFERRED_SIZE, 77, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(30, 30, 30))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void Button1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_Button1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_Button1ActionPerformed

    private void Button2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_Button2ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_Button2ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Game2048.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Game2048.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Game2048.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Game2048.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Game2048().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton Button1;
    private javax.swing.JButton Button10;
    private javax.swing.JButton Button11;
    private javax.swing.JButton Button12;
    private javax.swing.JButton Button13;
    private javax.swing.JButton Button14;
    private javax.swing.JButton Button15;
    private javax.swing.JButton Button16;
    private javax.swing.JButton Button2;
    private javax.swing.JButton Button3;
    private javax.swing.JButton Button4;
    private javax.swing.JButton Button5;
    private javax.swing.JButton Button6;
    private javax.swing.JButton Button7;
    private javax.swing.JButton Button8;
    private javax.swing.JButton Button9;
    private javax.swing.JTextField ScoreField;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel tips;
    // End of variables declaration//GEN-END:variables

  
}
