# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_form.ui'
#
# Created by: PyQt5 UI code generator 5.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import pymysql

class Ui_MainWindow(object):
    def login(self):
        username = self.lineEdit.text()
        age = self.lineEdit_2.text()
        gender = self.comboBox.currentText()
        degree = self.comboBox_2.currentText()
        conn = pymysql.connect(host='localhost',user='root',password='',db='mtp_subjects')
        cur = conn.cursor()
        query = 'select * from subject_info where name=%s'
        data = cur.execute(query,(username))
        if(len(cur.fetchall())>0):
            query = 'select session_number from subject_info where name=%s'
            data = cur.execute(query,(username))
            sess_num = cur.fetchone()
            query = 'update subject_info set session_number=%s where name=%s'
            data = cur.execute(query,(sess_num[0]+1,username))
            conn.commit()
            import os
            os.system('xdg-open '+ '/home/knot/Pictures/im{}.png'.format(sess_num[0]+1))
        else:
            query = 'select count(*) from subject_info'
            data = cur.execute(query)
            id_ = cur.fetchone()
            query = 'insert into subject_info(ID,Name,Age,Gender,Degree,session_number) values(%s,%s,%s,%s,%s,%s)'
            data = cur.execute(query,(id_, username, age, gender, degree,1))
            conn.commit()
            import os
            os.system('xdg-open '+ '/home/knot/Pictures/im{}.png'.format(1))



    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 120, 54, 14))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(160, 160, 54, 14))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(390, 330, 88, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.login)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 200, 54, 14))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(160, 240, 71, 16))
        self.label_4.setObjectName("label_4")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(300, 110, 113, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(300, 150, 113, 30))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(300, 190, 97, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Male")
        self.comboBox.addItem("Female")
        self.comboBox.addItem("Other")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(300, 230, 97, 30))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("UnderGraduate")
        self.comboBox_2.addItem("Graduate")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Name"))
        self.label_2.setText(_translate("MainWindow", "Age"))
        self.pushButton.setText(_translate("MainWindow", "Continue"))
        self.label_3.setText(_translate("MainWindow", "Gender"))
        self.label_4.setText(_translate("MainWindow", "Qualification"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Male"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Female"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Other"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
