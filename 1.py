import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showerror, showinfo

from PIL import Image, ImageTk
import cv2

from tkinter.ttk import Treeview, Scrollbar

from ultralytics import YOLO
from collections import Counter
import time
import os
import numpy as np

class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title('高精度行人车辆检测与追踪系统')
        self.geometry('800x600+300+100')
        self.resizable(False, False)
        self.config(bg='#87CEEB')

        self.is_mark = False
        self.result_list = []       # 每张图的检测结果缓存
        self.current_img_bgr = None # 当前原始cv图
        self.current_img_rgb = None
        self.img_show_w = 310
        self.img_show_h = 310

        self.initUI()
        self.init_model()

        # 绑定按钮事件
        self.openImageBtn.config(command=self.open_image)
        self.openDirBtn.config(command=self.open_dir)
        self.openVideoBtn.config(command=self.open_video)
        self.openCameraBtn.config(command=self.open_camera)
        self.saveBtn.config(command=self.save_result_img)
        self.exitBtn.config(command=self.quit)

        # 视频/摄像头控制标记
        self.cap = None
        self.video_running = False

    def initUI(self):
        self.lblTitle = Label(self, text='高精度行人车辆检测与追踪系统', font=("宋体", 20), bg='#87CEEB')
        self.lblTitle.pack(pady=20)

        self.line = Frame(self, width=780, height=1, bg='#B1B1B1')
        self.line.pack()

        self.mainFrame = Frame(self, bg='#87CEEB')
        self.mainFrame.pack(pady=10)

        # 图像显示Label
        self.photo = self.get_image_icon('res/ui_images/11.png', self.img_show_w, self.img_show_h)
        self.lblImage = Label(self.mainFrame, width=self.img_show_w + 170, height=self.img_show_h,
                              image=self.photo, bg='#FFFFFF')
        self.lblImage.pack(side=LEFT, padx=5)

        self.mainRightFrame = Frame(self.mainFrame, width=280, height=200, bg="#87CEEB")
        self.mainRightFrame.pack(side=LEFT)

        # 参数设置区
        self.paramFrame = LabelFrame(self.mainRightFrame, width=280, height=70, text='参数设置', bg='#87CEEB')
        self.paramFrame.pack(padx=5)
        self.paramFrame.grid_propagate(False)

        self.lblTrust = Label(self.paramFrame, text='置信度阈值', bg='#87CEEB')
        self.lblTrust.grid(row=0, column=0, pady=5)
        self.trustVar = DoubleVar()
        self.trustVar.set(0.30)
        self.trustBox = Spinbox(self.paramFrame, from_=0, to=1, textvariable=self.trustVar, increment=0.01, width=5)
        self.trustBox.grid(row=0, column=1, pady=5)

        self.lblComp = Label(self.paramFrame, text='交并比阈值', bg='#87CEEB')
        self.lblComp.grid(row=0, column=2, pady=5)
        self.CompVar = DoubleVar()
        self.CompVar.set(0.70)
        self.CompBox = Spinbox(self.paramFrame, from_=0, to=1, textvariable=self.CompVar, increment=0.01, width=5)
        self.CompBox.grid(row=0, column=3, pady=5)

        # 检测结果面板
        self.resultFrame = LabelFrame(self.mainRightFrame, width=280, height=250, text="检测结果", bg="#87CEEB")
        self.resultFrame.pack(padx=5)
        self.resultFrame.grid_propagate(False)

        self.lblResult1 = Label(self.resultFrame, text="目标总数：", bg="#87CEEB")
        self.lblResult1.grid(row=0, column=0, pady=5)
        self.lblResult2 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#E20000")
        self.lblResult2.grid(row=0, column=1, pady=5, sticky="w")

        self.lblResult3 = Label(self.resultFrame, text="    用时：", bg="#87CEEB")
        self.lblResult3.grid(row=0, column=2, pady=5, sticky="e")
        self.lblResult4 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#E20000")
        self.lblResult4.grid(row=0, column=3, pady=5, sticky="w")

        self.lblResult5 = Label(self.resultFrame, text="行人数目：", bg="#87CEEB")
        self.lblResult5.grid(row=1, column=0, pady=5)
        self.lblResult6 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#9800F0")
        self.lblResult6.grid(row=1, column=1, pady=5, sticky="w")

        self.lblResult7 = Label(self.resultFrame, text="    车辆数目：", bg="#87CEEB")
        self.lblResult7.grid(row=1, column=2, pady=5)
        self.lblResult8 = Label(self.resultFrame, text="", bg="#87CEEB", foreground="#9800F0")
        self.lblResult8.grid(row=1, column=3, pady=5, sticky="w")

        self.lineResult = Frame(self.resultFrame, width=260, height=1, bg='#B1B1B1')
        self.lineResult.grid(row=2, column=0, columnspan=4, padx=5)

        self.lblResult9 = Label(self.resultFrame, text='    目标选择：', bg='#87CEEB')
        self.lblResult9.grid(row=3, column=0, pady=5, sticky='w')

        self.targetlist = []
        self.cbTarget = ttk.Combobox(self.resultFrame, value=self.targetlist, width=10)
        self.cbTarget.grid(row=3, column=1, columnspan=3, pady=5, sticky='w')

        self.lblResult10 = Label(self.resultFrame, text='    类型：', bg='#87CEEB')
        self.lblResult10.grid(row=4, column=0, pady=5)
        self.lblResult11 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult11.grid(row=4, column=1, pady=5, sticky='w')

        self.lblResult12 = Label(self.resultFrame, text='    置信度：', bg='#87CEEB')
        self.lblResult12.grid(row=4, column=2, pady=5)
        self.lblResult13 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult13.grid(row=4, column=3, pady=5, sticky='w')

        self.lblResult14 = Label(self.resultFrame, text='     目标位置：', bg='#87CEEB')
        self.lblResult14.grid(row=5, column=0, columnspan=4, pady=5, sticky='w')

        self.lblResult15 = Label(self.resultFrame, text='    xmin：', bg='#87CEEB')
        self.lblResult15.grid(row=6, column=0, pady=5)
        self.lblResult16 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult16.grid(row=6, column=1, pady=5, sticky='w')

        self.lblResult17 = Label(self.resultFrame, text='    ymin：', bg='#87CEEB')
        self.lblResult17.grid(row=6, column=2, pady=5)
        self.lblResult18 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult18.grid(row=6, column=3, pady=5, sticky='w')

        self.lblResult19 = Label(self.resultFrame, text='    xmax：', bg='#87CEEB')
        self.lblResult19.grid(row=7, column=0, pady=5)
        self.lblResult20 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult20.grid(row=7, column=1, pady=5, sticky='w')

        self.lblResult21 = Label(self.resultFrame, text='    ymax：', bg='#87CEEB')
        self.lblResult21.grid(row=7, column=2, pady=5)
        self.lblResult22 = Label(self.resultFrame, text='', bg='#87CEEB', foreground='#E20D00')
        self.lblResult22.grid(row=7, column=3, pady=5, sticky='w')

        # 底部区域
        self.bottomFrame = Frame(self, width=770, height=160, bg='#87CEEB')
        self.bottomFrame.pack()
        self.bottomFrame.pack_propagate(False)

        self.tableFrame = LabelFrame(self.bottomFrame, width=485, height=160, text='识别结果信息', bg='#87CEEB')
        self.tableFrame.pack(side=LEFT)
        self.tableFrame.pack_propagate(False)

        self.result_table = ttk.Treeview(self.tableFrame, columns=('id', 'path', 'track', 'type', 'trust', 'pos'), show='headings')
        self.result_table.heading('id', text='序号')
        self.result_table.heading('path', text='文件路径')
        self.result_table.heading('track', text='追踪ID')
        self.result_table.heading('type', text='类型')
        self.result_table.heading('trust', text='置信度')
        self.result_table.heading('pos', text='坐标位置')

        self.result_table.column('id', width=50, anchor='center')
        self.result_table.column('path', width=100, anchor='w')
        self.result_table.column('track', width=60, anchor='center')
        self.result_table.column('type', width=60, anchor='center')
        self.result_table.column('trust', width=60, anchor='center')
        self.result_table.column('pos', width=150, anchor='center')

        self.result_table.pack(fill='both', expand=True)
        self.rightScrollBar = Scrollbar(self.tableFrame, orient="vertical", command=self.result_table.yview)
        self.rightScrollBar.pack(fill=Y, side=RIGHT)
        self.result_table.config(yscrollcommand=self.rightScrollBar.set)

        # 操作面板
        self.optFrame = LabelFrame(self.bottomFrame, width=280, height=160, text='操作', bg='#87CEEB')
        self.optFrame.pack(side=RIGHT)
        self.optFrame.grid_propagate(False)

        self.openIcon = self.get_image_icon("res/ui_images/icons/img.png")
        self.openImageBtn = Button(self.optFrame, width=90, height=25, image=self.openIcon, compound=LEFT, text="打开图片", bg="#FECD9A", borderwidth=0)
        self.openImageBtn.grid(row=0, column=0, padx=20, pady=10)

        self.openDirIcon = self.get_image_icon("res/ui_images/icons/folder.png")
        self.openDirBtn = Button(self.optFrame, width=90, height=25, image=self.openDirIcon, compound=LEFT, text="打开文件夹", bg="#FECD9A", borderwidth=0)
        self.openDirBtn.grid(row=0, column=1)

        self.openVideoIcon = self.get_image_icon("res/ui_images/icons/video.png")
        self.openVideoBtn = Button(self.optFrame, width=90, height=25, image=self.openVideoIcon, compound=LEFT, text="打开视频", bg="#FECD9A", borderwidth=0)
        self.openVideoBtn.grid(row=1, column=0)

        self.openCameraIcon = self.get_image_icon("res/ui_images/icons/camera.png")
        self.openCameraBtn = Button(self.optFrame, width=90, height=25, image=self.openCameraIcon, compound=LEFT, text="打开摄像头", bg="#FECD9A", borderwidth=0)
        self.openCameraBtn.grid(row=1, column=1)

        self.saveIcon = self.get_image_icon("res/ui_images/icons/save.png")
        self.saveBtn = Button(self.optFrame, width=90, height=25, image=self.saveIcon, compound=LEFT, text="保存", bg="#FECD9A", borderwidth=0)
        self.saveBtn.grid(row=2, column=0, pady=10)

        self.exitIcon = self.get_image_icon("res/ui_images/icons/exit.png")
        self.exitBtn = Button(self.optFrame, width=90, height=25, image=self.exitIcon, compound=LEFT, text="退出", bg="#FECD9A", borderwidth=0)
        self.exitBtn.grid(row=2, column=1, pady=10)

        # 事件绑定
        self.cbTarget.bind("<<ComboboxSelected>>", self.change_target)
        self.result_table.bind('<ButtonRelease-1>', self.tree_click)

    def init_model(self):
        try:
            self.model = YOLO("models/best.pt", task='detect')
        except Exception as e:
            showerror("模型加载失败", f"请检查 models/best.pt 是否存在！\n{e}")
            self.model = None

    def get_image_icon(self, path, width=20, height=20):
        try:
            image = Image.open(path)
            image = image.resize((width, height))
            photo = ImageTk.PhotoImage(image)
            return photo
        except:
            return None

    def draw_result_on_img(self, img_bgr, result, select_idx=-1):
        """在图上画框，选中目标会高亮红色，其他绿色"""
        img = img_bgr.copy()
        boxes = result.boxes
        if boxes is None:
            return img
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy()
        else:
            track_ids = [-1]*len(xyxy)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)
            c = int(clses[i])
            conf = confs[i]
            tid = int(track_ids[i])
            label = f"{'person' if c==0 else 'car'} {conf:.2f} ID:{tid}"
            if i == select_idx:
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
            else:
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        return img

    def show_img_to_label(self, img_bgr):
        """把opencv图缩放到固定大小，显示在tk label"""
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((self.img_show_w, self.img_show_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.lblImage.config(image=self.photo)

    def mark_image(self, img_bgr, use_track=False):
        t1 = time.time()
        conf = round(self.trustVar.get(), 2)
        iou = round(self.CompVar.get(), 2)
        if use_track:
            results = self.model.track(img_bgr, conf=conf, iou=iou, persist=True)[0]
        else:
            results = self.model(img_bgr, conf=conf, iou=iou)[0]
        t2 = time.time()
        cost = "{:.3f} s".format(t2 - t1)
        self.result_list.clear()
        self.result_list.append({"results": results, "taken_time": cost})
        self.current_img_bgr = img_bgr
        self.show_window_info(0)

    def show_window_info(self, index=0):
        if len(self.result_list) == 0:
            return
        item = self.result_list[index]
        res = item["results"]
        cost = item["taken_time"]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            self.lblResult2.config(text="0")
            self.lblResult4.config(text=cost)
            self.lblResult6.config(text="0")
            self.lblResult8.config(text="0")
            self.cbTarget.config(values=["无目标"])
            self.cbTarget.current(0)
            self.show_img_to_label(self.current_img_bgr)
            self.clear_detail_info()
            return

        xyxy_list = boxes.xyxy.cpu().numpy().astype(int).tolist()
        cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
        conf_list = boxes.conf.cpu().numpy().tolist()
        if boxes.id is not None:
            tid_list = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            tid_list = [-1]*len(cls_list)

        # 统计
        cnt = Counter(cls_list)
        total = len(cls_list)
        self.lblResult2.config(text=str(total))
        self.lblResult4.config(text=cost)
        self.lblResult6.config(text=str(cnt.get(0, 0)))
        self.lblResult8.config(text=str(cnt.get(1, 0)))

        # 下拉框
        target_options = ["全部"]
        for idx, c in enumerate(cls_list):
            name = "person" if c == 0 else "car"
            target_options.append(f"{name}_{idx}")
        self.cbTarget.config(values=target_options)
        self.cbTarget.current(0)

        # 清空表格，写入新数据
        for row in self.result_table.get_children():
            self.result_table.delete(row)
        for idx, (xyxy, c, conf, tid) in enumerate(zip(xyxy_list, cls_list, conf_list, tid_list)):
            typ = "行人" if c ==0 else "车辆"
            pos_text = f"{xyxy[0]},{xyxy[1]},{xyxy[2]},{xyxy[3]}"
            self.result_table.insert("", END, values=(idx+1, "当前图", tid, typ, f"{conf:.2f}", pos_text))

        # 默认绘制全部框
        draw_img = self.draw_result_on_img(self.current_img_bgr, res, select_idx=-1)
        self.show_img_to_label(draw_img)
        self.clear_detail_info()

    def clear_detail_info(self):
        self.lblResult11.config(text="")
        self.lblResult13.config(text="")
        self.lblResult16.config(text="")
        self.lblResult18.config(text="")
        self.lblResult20.config(text="")
        self.lblResult22.config(text="")

    def change_target(self, event):
        val = self.cbTarget.get()
        if val == "全部" or len(self.result_list) == 0:
            self.show_window_info(0)
            return
        _, idx_str = val.split("_")
        sel_idx = int(idx_str)
        res = self.result_list[0]["results"]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        x1,y1,x2,y2 = xyxy[sel_idx]
        c = cls[sel_idx]
        cf = conf[sel_idx]
        typ = "行人" if c == 0 else "车辆"
        self.lblResult11.config(text=typ)
        self.lblResult13.config(text=f"{cf*100:.2f}%")
        self.lblResult16.config(text=str(x1))
        self.lblResult18.config(text=str(y1))
        self.lblResult20.config(text=str(x2))
        self.lblResult22.config(text=str(y2))

        # 选中目标高亮
        draw_img = self.draw_result_on_img(self.current_img_bgr, res, select_idx=sel_idx)
        self.show_img_to_label(draw_img)

    def tree_click(self, event):
        sel = self.result_table.selection()
        if not sel:
            return
        row = self.result_table.item(sel[0], "values")
        idx = int(row[0]) - 1
        cls_name = row[3]
        self.cbTarget.set(f"{'person' if cls_name=='行人' else 'car'}_{idx}")
        self.change_target(None)

    def open_image(self):
        fp = filedialog.askopenfilename(filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not fp:
            return
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            showerror("错误", "图片读取失败")
            return
        self.mark_image(img, use_track=False)

    def open_dir(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        img_suffix = (".jpg", ".jpeg", ".png", ".bmp")
        img_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(img_suffix)]
        if len(img_list) == 0:
            showinfo("提示", "文件夹内没有图片")
            return
        # 只演示第一张，你也可以加翻页
        fp = img_list[0]
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.mark_image(img, use_track=False)
        showinfo("批量提示", f"一共找到 {len(img_list)} 张图片，当前展示第一张，可自行扩展翻页")

    def open_video(self):
        if self.video_running:
            self.video_running = False
            self.cap.release()
            return
        fp = filedialog.askopenfilename(filetypes=[("视频", "*.mp4;*.avi;*.mov;*.mkv")])
        if not fp:
            return
        self.cap = cv2.VideoCapture(fp)
        self.video_running = True
        self.video_loop()

    def open_camera(self):
        if self.video_running:
            self.video_running = False
            self.cap.release()
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            showerror("错误", "摄像头打开失败")
            return
        self.video_running = True
        self.video_loop()

    def video_loop(self):
        if not self.video_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.video_running = False
            self.cap.release()
            return
        self.current_img_bgr = frame
        self.mark_image(frame, use_track=True)
        self.after(10, self.video_loop)

    def save_result_img(self):
        if self.current_img_bgr is None:
            showerror("提示", "没有可保存图像")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("jpg图片", "*.jpg"), ("png图片", "*.png")])
        if not save_path:
            return
        cv2.imencode('.jpg', self.current_img_bgr)[1].tofile(save_path)
        showinfo("成功", "检测图保存完成！")

app = Application()
app.mainloop()
