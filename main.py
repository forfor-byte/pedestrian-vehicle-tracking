import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showerror

from PIL import Image , ImageTk
import cv2

from tkinter.ttk import Treeview, Scrollbar

from ultralytics import YOLO
from collections import Counter
import time
import os
import numpy as np

class Application(Tk):
    # 构造函数
    def __init__(self):
        Tk.__init__(self)
        self.title('高精度行人车辆检测与追踪系统')
        self.geometry('800x600+300+100')
        # 窗口限定 false: 宽度不变 false: 高度不变
        self.resizable(False , False)
        self.config(bg='#87CEEB')

        self.is_mark = False
        # 存储所有图片的标注对象
        self.result_list = []
        # 初始化ui界面
        self.initUI()
        # 加载模型
        self.init_model()

    def initUI(self):
        # 标题文本
        self.lblTitle = Label(self, text='高精度行人车辆检测与追踪系统', font=("宋体", 20), bg='#87CEEB')
        self.lblTitle.pack(pady=20)
        # 水平线
        self.line = Frame(self, width=780, height=1, bg='#B1B1B1')
        self.line.pack()
        # 中间区域内容
        self.mainFrame = Frame(self, bg='#87CEEB')
        self.mainFrame.pack(pady=10)
        # 显示图片
        imageW = 310
        imageH = 310
        self.photo = self.get_image_icon('res/ui_images/11.png', imageW, imageH)
        self.lblImage = Label(self.mainFrame, width=imageW + 170, height=imageH, image=self.photo, bg='#FFFFFF')
        self.lblImage.pack(side=LEFT , padx = 5 )
        # 中间区域右侧显示内容
        self.mainRightFrame = Frame(self.mainFrame, width=280, height=200, bg="#87CEEB")
        self.mainRightFrame.pack(side=LEFT)

        #参数设置区域：
        self.paramFrame = LabelFrame(self.mainRightFrame, width=280, height=70, text='参数设置', bg='#87CEEB')
        self.paramFrame.pack(padx=5)
        self.paramFrame.grid_propagate(False)

        # 置信度阈值 - 输入控件
        self.lblTrust = Label(self.paramFrame , text = '置信度阈值' , bg = '#87CEEB')
        self.lblTrust.grid(row = 0 , column= 0 , pady = 5)
        # 定义变量用于和输入框进行绑定
        self.trustVar = DoubleVar()
        self.trustVar.set(0.30)
        self.trustBox = Spinbox(self.paramFrame , from_=0 , to=1 , textvariable=self.trustVar , increment=0.01 , width=5)
        self.trustBox.grid(row=0 , column=1 , pady=5)

        # 交并比阈值 - 输入控件 self.lblComp 放在置信度右侧
        self.lblComp = Label(self.paramFrame, text='交并比阈值', bg='#87CEEB')
        self.lblComp.grid(row=0, column=2, pady=5)
        # 定义变量用于和输入框进行绑定
        self.CompVar = DoubleVar()
        self.CompVar.set(0.70)
        self.CompBax = Spinbox(self.paramFrame, from_=0, to=1, textvariable=self.CompVar, increment=0.01, width=5)
        self.CompBax.grid(row=0, column=3, pady=5)

        ## 检测结果区域
        self.resultFrame = LabelFrame(self.mainRightFrame, width=280, height=250, text="检测结果", bg="#87CEEB")
        self.resultFrame.pack(padx=5)
        self.resultFrame.grid_propagate(False)

        # 目标总数文本
        self.lblResult1 = Label(self.resultFrame, text="目标总数：", bg="#87CEEB")
        self.lblResult1.grid(row=0, column=0, pady=5)
        # 目标总数结果
        self.lblResult2 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#E20000")
        self.lblResult2.grid(row=0, column=1, pady=5, sticky="w")

        # 用时文本
        self.lblResult3 = Label(self.resultFrame, text="    用时：", bg="#87CEEB")
        self.lblResult3.grid(row=0, column=2, pady=5, sticky="e")
        # 用时结果
        self.lblResult4 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#E20000")
        self.lblResult4.grid(row=0, column=3, pady=5, sticky="w")

        # 行人数目
        self.lblResult5 = Label(self.resultFrame, text="行人数目：", bg="#87CEEB")
        self.lblResult5.grid(row=1, column=0, pady=5)
        # 行人数目结果
        self.lblResult6 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#9800F0")
        self.lblResult6.grid(row=1, column=1, pady=5, sticky="w")

        # 车辆数目
        self.lblResult7 = Label(self.resultFrame, text="    车辆数目：", bg="#87CEEB")
        self.lblResult7.grid(row=1, column=2, pady=5)

        #水平线
        self.lineResult = Frame(self.resultFrame , width = 260 , height = 1 , bg = '#B1B1B1')
        self.lineResult.grid(row = 2 , column = 0 , columnspan = 4, padx = 5)

        # 目标选择
        self.lblResult9 = Label(self.resultFrame, text = '    目标选择：' , bg='#87CEEB')
        self.lblResult9.grid(row=3, column=0,  pady=5 , sticky = 'w')

        # 目标选择框
        self.targetlist = []
        self.cbTarget = ttk.Combobox(self.resultFrame , value = self.targetlist , width = 10)
        self.cbTarget.grid(row = 3 , column = 1 , columnspan = 3 , pady = 5 , sticky = 'w')

        # 类型
        self.lblResult10 = Label(self.resultFrame, text='    类型：', bg='#87CEEB')
        self.lblResult10.grid(row=4, column=0, pady=5)

        # 类型结果
        self.lblResult11 = Label(self.resultFrame, text='', bg='#87CEEB' , foreground = '#E20D00')
        self.lblResult11.grid(row=4, column=1, pady=5 , sticky = 'w')

        # 置信度
        self.lblResult12 = Label(self.resultFrame, text='    置信度：', bg='#87CEEB')
        self.lblResult12.grid(row=4, column=2, pady=5)

        # 置信度结果
        self.lblResult13 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult13.grid(row=4, column=3, pady=5, sticky='w')

        # 目标位置
        self.lblResult14 = Label(self.resultFrame, text='     目标位置：', bg='#87CEEB')
        self.lblResult14.grid(row=5, column=0,columnspan = 4 , pady=5, sticky='w')

        # xmin
        self.lblResult15 = Label(self.resultFrame, text='    xmin：', bg='#87CEEB')
        self.lblResult15.grid(row=6, column=0, pady=5)

        # xmin结果
        self.lblResult16 = Label(self.resultFrame, text='', bg='#87CEEB' , foreground='#E20D00')
        self.lblResult16.grid(row=6, column=1, pady=5, sticky = 'w')

        # ymin
        self.lblResult17 = Label(self.resultFrame, text='    ymin：', bg='#87CEEB')
        self.lblResult17.grid(row=6, column=2, pady=5)

        # ymin结果
        self.lblResult18 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult18.grid(row=6, column=3, pady=5, sticky='w')

        # xmax
        self.lblResult19 = Label(self.resultFrame, text='    xmax：', bg='#87CEEB')
        self.lblResult19.grid(row=7, column=0, pady=5)

        # xmax结果
        self.lblResult20 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult20.grid(row=7, column=1, pady=5, sticky='w')

        # ymax
        self.lblResult21 = Label(self.resultFrame, text='    ymax：', bg='#87CEEB')
        self.lblResult21.grid(row=7, column=2, pady=5)

        # ymax结果
        self.lblResult22 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult22.grid(row=7, column=3, pady=5, sticky='w')

        # 底部区域
        self.bottomFrame = Frame(self , width = 770 , height = 160 , bg = '#87CEEB')
        self.bottomFrame.pack()
        # 保证宽高起作用
        self.bottomFrame.pack_propagate(False)

        # 表格区域
        self.tableFrame = LabelFrame(self.bottomFrame , width = 485 , height = 160 , text ='识别结果信息' , bg = '#87CEEB')
        self.tableFrame.pack(side = LEFT)
        self.tableFrame.pack_propagate(False)

        self.result_table = ttk.Treeview(self.tableFrame, columns=('id', 'path', 'track', 'type' , 'trust' , 'pos'), show='headings')

        # 3. 设置表头（列名）
        self.result_table.heading('id', text='序号')
        self.result_table.heading('path', text='文件路径')
        self.result_table.heading('track', text='追踪')
        self.result_table.heading('type', text='类型')
        self.result_table.heading('trust', text='置信度')
        self.result_table.heading('pos', text='坐标位置')

        # 4. 设置每列宽度
        self.result_table.column('id', width=50, anchor='center')  # anchor可选，设置内容居中
        self.result_table.column('path', width=100, anchor='w')  # 左对齐，适合长文本
        self.result_table.column('track', width=50, anchor='center')
        self.result_table.column('type', width=50, anchor='center')
        self.result_table.column('trust', width=50, anchor='center')
        self.result_table.column('pos', width=150, anchor='center')

        # 5. 用pack方式摆放表格，填充整个容器
        self.result_table.pack(fill='both', expand=True)

        self.rightScrollBar = Scrollbar(self.tableFrame, orient="vertical", command=self.result_table.yview)
        self.rightScrollBar.pack(fill=Y, side=RIGHT)
        self.result_table.config(yscrollcommand=self.rightScrollBar.set)

        # 底部，右侧操作显示区域
        self.optFrame = LabelFrame(self.bottomFrame, width=280, height=160, text='操作' , bg = '#87CEEB')
        self.optFrame.pack(side=RIGHT)
        self.optFrame.grid_propagate(False)  # 保证宽高起作用
        # 操作显示区域-打开图片
        self.openIcon = self.get_image_icon("res/ui_images/icons/img.png")
        self.openImageBtn = Button(self.optFrame, width=90, height=25,image=self.openIcon, compound=LEFT, text="打开图片", bg="#FECD9A",borderwidth=0)
        self.openImageBtn.grid(row=0, column=0, padx=20, pady=10)
        # 操作显示区域-打开文件夹
        self.openDirIcon = self.get_image_icon("res/ui_images/icons/folder.png")
        self.openDirBtn = Button(self.optFrame, width=90, height=25,image=self.openDirIcon, compound=LEFT, text="打开文件夹", bg="#FECD9A",borderwidth=0)
        self.openDirBtn.grid(row=0, column=1)
        # 操作显示区域-打开视频
        self.openVideoIcon = self.get_image_icon("res/ui_images/icons/video.png")
        self.openVideoBtn = Button(self.optFrame, width=90, height=25,image=self.openVideoIcon, compound=LEFT, text="打开视频", bg="#FECD9A",borderwidth=0)
        self.openVideoBtn.grid(row=1, column=0)
        # 操作显示区域-打开摄像头
        self.openCameraIcon = self.get_image_icon("res/ui_images/icons/camera.png")
        self.openCameraBtn = Button(self.optFrame, width=90, height=25,image=self.openCameraIcon, compound=LEFT, text="打开摄像头",bg="#FECD9A", borderwidth=0)
        self.openCameraBtn.grid(row=1, column=1)
        # 操作显示区域-保存

        self.saveIcon = self.get_image_icon("res/ui_images/icons/save.png")
        self.saveBtn = Button(self.optFrame, width=90, height=25,image=self.saveIcon, compound=LEFT, text="保存", bg="#FECD9A",borderwidth=0)
        self.saveBtn.grid(row=2, column=0, pady=10)
        # 操作显示区域-退出
        self.exitIcon = self.get_image_icon("res/ui_images/icons/exit.png")
        self.exitBtn = Button(self.optFrame, width=90, height=25, image=self.exitIcon,compound=LEFT, text="退出", bg="#FECD9A", borderwidth=0)
        self.exitBtn.grid(row=2, column=1, pady=10)

        # 目标选址下拉框时间
        self.cbTarget.bind("<<ComboboxSelected>>" , self.change_target)
        self.result_table.bind('<ButtonRelease-1>' , self.tree_click)



    # 加载模型
    def init_model(self):
        # 加载检测模型
        self.model = YOLO("models/best.pt", task='detect')

    # 获得图片图标，并更改大小
    def get_image_icon(self , path , width = 20 , height = 20):
        image = Image.open(path)
        # 获取图片原始的宽度和高度
        oldWidth = image.width
        oldHeight = image.height
        # 变换宽度和高度
        newWidth = int(oldWidth * (height / oldHeight))
        newHeight = int(height)

        # 更改图片大小
        self.image = image.resize((newWidth, newHeight))
        # 转换为tk可以使用的格式
        phone = ImageTk.PhotoImage(self.image)
        return phone

    #标注图像
    def mark_image(self , img):
        # 记录起始时间
        t1 = time.time()
        # 获得置信度阈值、交并比阈值
        self.conf = round(self.trustVar.get(), 2)
        self.iou = round(self.compVar.get(), 2)
        # 目标检测；参数1：图片路径、conf置信度阈值、iou交并比阈值

         # self.results=self.model.track(filePath, conf=0.30, iou=0.70)[0]
         # 目标检测；参数1：图像、conf置信度阈值、iou交并比阈值
        self.results = self.model(img, conf=self.conf, iou=self.iou)[0]
         # 记录结束时间
        t2 = time.time()
        # 将标注对象存储到列表中
        dict = {
        "results": self.results,
        "taken_time": "{:.3f} s".format(t2 - t1)
    }
        self.obj = dict["results"]  # 标注对象


        # 获得图片中所有标注对象的信息
        location_list = self.obj.boxes.xyxy.tolist()  # 获得所有标注框坐标
        self.location_list = [list(map(int, e)) for e in location_list]  # 将标注框坐标转换为整数
        cls_list = self.obj.boxes.cls.tolist()  # 标注内容是“人0”还是“车1”

        # 设置窗口显示信息[index：多个标注对象的索引]
        self.cls_list = [int(i) for i in cls_list]  # 将标注内容分类转换为整数
        self.conf_list = self.obj.boxes.conf.tolist()  # 获得标注内容的“置信度”
        self.conf_list = ['%.2f%%' % (each * 100) for each in self.conf_list]  #将置信度转换为百分比值
        # 显示窗口图片
        self.show_image(-1, self.obj)

    def show_window_info(self , index = 0):
        # 获得标注对象
        dict = self.result_list[index]
        self.obj = dict["results"]  # 标注对象

        # 获得图片中所有标注对象的信息
        location_list = self.obj.boxes.xyxy.tolist()  # 获得所有标注框坐标
        self.location_list = [list(map(int, e)) for e in location_list]  # 将标注框坐标转换为整数
        cls_list = self.obj.boxes.cls.tolist()  # 标注内容是“人0”还是“车1”
        self.cls_list = [int(i) for i in cls_list]  # 将标注内容分类转换为整数


        self.conf_list = self.obj.boxes.conf.tolist()  # 获得标注内容的“置信度”
        self.conf_list = ['%.2f%%' % (each * 100) for each in self.conf_list]  #将置信度转换为百分比值
        # 设置目标总数
        self.lblResult2.config(text=str(len(self.cls_list)))
        # 设置用时
        take_time = dict["taken_time"]
        self.lblResult4.config(text=take_time)
        # 行人数目、车辆数目
        take_count = Counter(self.cls_list)
        self.lblResult6.config(text=str(take_count[0]))
        self.lblResult8.config(text=str(take_count[1]))
        # 目标选择下拉框
        self.targetList.clear()
        self.targetList.append("全部")
        for k, v in enumerate(self.cls_list):
            if v == 1:
                self.targetList.append("car_" + str(k))
            else:
                self.targetList.append("person_" + str(k))
                self.cbTarget.config(values=self.targetList)
                self.cbTarget.current(0)
                # 显示标注图片及相关信息
            self.show_image(-1, self.obj)


    # 打开图片
    def open_image(self):
        # 文件选择对话框
        self.filePath = filedialog.askopenfilename(filetypes=[("打开文件",
                                                               "*.png"), ("打开文件", "*.jpg"), ("打开文件", "*.gif"),
                                                              ("打开文件", "*.jpeg")])
        if self.filePath:
        # 清空表格
            for item in self.tree.get_children():
                self.tree.delete(item)
                # 清空标注对象列表
            self.result_list.clear()
            #读取图片
            img = cv2.imdecode(np.fromfile(self.filePath , dtype=np.uint8) ,cv2.IMREAD_COLOR)
            # 检测
            self.mark_image(img)

    # 设置显示的图片及信息
    def show_image(self , index = 1 , obj = None):
        idx = index
        if index == -1:
            idx = 0
        # 结果信息
        if len(self.cls_list) > 0:
            if self.cls_list[idx] == 1:
                self.lblResult11.config(text = '汽车')
            else:
                self.lblResult11.config(text = '行人')
            self.lblResult13.config(text = str(self.conf_list[idx]))
            self.lblResult16.config(text=str(self.location_list[0]))
            self.lblResult18.config(text=str(self.location_list[1]))
            self.lblResult20.config(text=str(self.location_list[2]))
            self.lblResult22.config(text=str(self.location_list[3]))

            # 打开图片按钮
    def change_target(self):
        pass

    # 打开文件夹
    def tree_click(self):
       pass

    # 打开视频按钮
    def open_video(self):
       pass
    # 打开摄像头
    def open_camera(self):
        pass


app = Application()
app.mainloop()
