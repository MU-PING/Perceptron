# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:36:02 2020
@author: MU-PING
"""

import matplotlib.pyplot as plt
import tkinter as tk
import os
import numpy as np
import time
from math import floor, ceil
from random import randint
from matplotlib import animation
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from collections import defaultdict

class allfunction():
    def __init__(self, filename):
        #資料相關參數
        self.filename=filename
        self.all_data=[]    #全部資料  
        self.train_data=[]  #訓練資料
        self.test_data=[]   #測試資料
        self.dimension=0    #維度
        self.all_count=0    #全部資料數
        self.train_count=0  #訓練資料數
        self.test_count=0   #測試資料數
        self.color_dict={0:'blue',1:'red',2:'orange',3:'green'}
        self.label=1                            #標籤種類
        self.label_checker = defaultdict(int)   #標籤映射
        
        #製圖相關參數
        self.all_data_transpose=[]  #全部資料的轉置(用於計算x y z軸最大小值 
        self.ax=None
        self.max_x=0
        self.min_x=0
        self.max_y=0
        self.min_y=0
        self.max_z=0
        self.min_z=0
        self.font_size=0 #點大小
        self.line=False #判斷是否訓練過
        self.plot=None
        
        #訓練相關參數
        self.weight=None
        self.bias=None
        self.epoch=1
        self.accuracy=0
        self.learning_rate=0

        self.readfile()
        self.basic_setting()

    #---讀檔、基礎設定、畫圖、
    def readfile(self): 
        with open(os.getcwd()+'/DataSet/'+ self.filename +".txt", 'r', encoding='UTF-8') as file:
            for data in file.readlines():
                each_data = data.replace("\n", "").split(" ")
                each_data = [float(i) for i in each_data]
                each_data[-1]=int(each_data[-1])
                if(self.label_checker[each_data[-1]]==0): #代表出現新標籤
                    self.label_checker[each_data[-1]]=self.label
                    self.label+=1
                each_data[-1]=self.label_checker[each_data[-1]]-1    
                self.all_count+=1
                self.all_data.append(each_data)
            
        self.all_data=np.asarray(self.all_data)        
        self.dimension = len(self.all_data[0])-1   
        self.label-=1
        
        #讀完資料設置製圖的最大x y z 軸
        self.all_data_transpose=self.all_data.transpose() #用於取x y (z) 軸最大值
        self.max_x = ceil(max(self.all_data_transpose[0])+0.3)
        self.min_x = floor(min(self.all_data_transpose[0])-0.3)
        self.max_y = ceil(max(self.all_data_transpose[1])+0.3)
        self.min_y = floor(min(self.all_data_transpose[1])-0.3)
        self.font_size=2 if int(30/self.all_count)<=0 else int(30/self.all_count)   #點大小
            
    def basic_setting(self):
        if(self.dimension>3):
            messagebox.showwarning("Warning","三維以上無法顯示")
        else:
            self.draw(self.all_data)
            
            
        if(self.label>2):
            messagebox.showwarning("Warning",str(self.label)+"類無法使用PLA分類")
            btn['state'] = tk.DISABLED
            btn1['state'] = tk.DISABLED
            btn2['state'] = tk.DISABLED
            btn3['state'] = tk.DISABLED
        else:
            self.classification()
            btn["state"]=tk.NORMAL
            btn1["state"]=tk.NORMAL
            btn2["state"]=tk.NORMAL
            btn3["state"]=tk.NORMAL
            btn.configure(command=self.startlearning)
            btn1.configure(command=lambda: self.draw(self.all_data, "All"))
            btn2.configure(command=lambda: self.draw(self.train_data, "Train"))
            btn3.configure(command=lambda: self.draw(self.test_data, "Test"))
            #初始化參數
        lr_display.set("")
        all_count_display.set("")
        train_count_display.set("")
        test_count_display.set("")
        weight_display.set("")
        dataset.set("")
        epoch_display.set("")
        single_display.set("")
        all_display.set("")
        train_accuracy_display.set("")
        test_accuracy_display.set("")
        
    def draw(self, data, title="All"):
        plt.clf() #清除圖片 參考：https://reurl.cc/9XnRrj
        plt.title(title)
        if(self.dimension==2):
            plt.xlim(self.min_x, self.max_x)
            plt.ylim(self.min_y, self.max_y)
            for i in data:
                plt.plot(i[0], i[1], 'o', ms=self.font_size , color = self.color_dict[int(i[-1]%self.label)]) #畫圖 ms：折點大小
            
            if(self.line): #如果訓練過則要畫出權重圖
                x = np.array(range(self.min_x, self.max_x+1))
                y = -(self.weight[0]*x + self.bias)/self.weight[1]
                plt.plot(x, y ,linewidth = 1, color = 'black') #畫圖 ms：折點大小

            canvas.draw()   
        elif(self.dimension==3):
            self.max_z = ceil(max(self.all_data_transpose[2]))
            self.min_z = floor(min(self.all_data_transpose[2]))
            self.ax = Axes3D(fig) #3D圖
            self.ax.set_xlim(self.min_x, self.max_x)
            self.ax.set_ylim(self.min_y, self.max_y)
            self.ax.set_zlim(self.min_z, self.max_z)
            for i in data:  #ax.plot版本不一樣語法不一樣
                self.ax.plot([i[0]], [i[1]], [i[2]], 'o', ms=self.font_size , color = self.color_dict[int(i[-1]%self.label)])
            
            if(self.line):
                x = np.array(range(self.min_x, self.max_x+1))
                y = np.array(range(self.min_y, self.max_y+1))
                X, Y = np.meshgrid(x, y)
                Z = -(self.weight[0]*X + self.weight[1]*Y + self.bias)/self.weight[2]
                self.ax.plot_surface(X, Y, Z, color = 'black', alpha=.5)
            canvas.draw()
        
    #---分割訓練、測試資料  
    def classification(self): 
        temp_train_count=0
        temp_test_count=0

        self.train_count=ceil(2*self.all_count/3) # #訓練資料數最多2/3 
        self.test_count=self.all_count-self.train_count   #測試料數
        for i in range(self.all_count):    #全部資料
            if (randint(0,1)>0.5):
                self.train_data.append(self.all_data[i])
                temp_train_count+=1
            else:
                self.test_data.append(self.all_data[i])
                temp_test_count+=1
            if(temp_train_count==self.train_count): #訓練資料集先達到2/3
                for k in range(i+1,self.all_count):
                    self.test_data.append(self.all_data[k])
                break
            elif(temp_test_count==self.test_count):
                for k in range(i+1,self.all_count):  #測試資料集先達到1/3
                    self.train_data.append(self.all_data[k])
                break
        
        self.train_data = np.asarray(self.train_data)
        self.test_data = np.asarray(self.test_data)

            
    #---PLA訓練、評價相關
    def startlearning(self):
        if(not self.check_learning_setting()): return #檢查基礎設定
        #關閉開始、選擇按鈕
        self.draw(self.train_data, "Train")
        btn['state'] = tk.DISABLED
        btn1['state'] = tk.DISABLED
        btn2['state'] = tk.DISABLED
        btn3['state'] = tk.DISABLED
        data_combobox['state'] = tk.DISABLED

        #設定資訊欄
        lr_display.set(str(self.learning_rate))
        all_count_display.set(str(self.all_count))
        train_count_display.set(str(self.train_count))
        test_count_display.set(str(self.test_count))
        all_display.set("/"+str(self.train_count))
        dataset.set(data_combobox.get())
        
        self.weight = np.random.rand(self.dimension) #產生權重
        self.bias = 0                               #偏移量
        self.line = True

        frames=self.gen_function_convergence1 if convergence.get()==0 else self.gen_function_convergence2 #判斷收斂條件
        

        # blit=True 可以自動移除上個圖形，繪製下個圖形，但似乎不支援Axes3D
        if(self.dimension==2):
            #https://reurl.cc/XkNW2j https://reurl.cc/147YMG 
            ani = animation.FuncAnimation(fig=fig, func=self.update_2D, frames=frames, init_func=lambda: self.init(self.dimension),
                                          interval=72/((update_combobox.current()+1)*(update_combobox.current()+1)), blit=False, repeat=False) #動畫

            canvas.draw()
        elif(self.dimension==3):
            ani = animation.FuncAnimation(fig=fig, func=self.update_3D, frames=frames, init_func=lambda: self.init(self.dimension),
                                          interval=72/((update_combobox.current()+1)*(update_combobox.current()+1)), blit=False, repeat=False) #動畫
            canvas.draw()
        else:
            ani = animation.FuncAnimation(fig=fig, func=self.update_up_4D, frames=frames, init_func=lambda: self.init(self.dimension),
                                          interval=72/((update_combobox.current()+1)*(update_combobox.current()+1)), blit=False, repeat=False) #動畫
            canvas.draw()
            
    # FuncAnimation如果沒有設定init_func，則func的第一禎會執行兩次: 第一次為初始化、第二次才正式開始
    # func會自動執行plt.draw()，不需要加入，init_func則不會
    def init(self, dimension): 
        
        if(dimension==2):
            x = np.array(range(self.min_x, self.max_x+1))
            y = -(self.weight[0]*x + self.bias)/self.weight[1]
            self.plot=plt.plot(x, y ,linewidth = 1, color = 'black') #畫圖 ms：折點大小
            plt.draw() #預防初始化時即達到收斂條件，圖形畫不出來
           
        elif(dimension==3):
            x = np.array(range(self.min_x, self.max_x+1))
            y = np.array(range(self.min_y, self.max_y+1))
            X, Y = np.meshgrid(x, y)
            Z = -(self.weight[0]*X + self.weight[1]*Y + self.bias)/self.weight[2]
            self.plot=[self.ax.plot_surface(X, Y, Z, color = 'black', alpha=.5)]
            plt.draw()
            
        else: #三維以上無法畫圖
            pass
            
    def update_2D(self, i): #2維資料更新參數
        temp=hardlim(self.train_data[i][0:-1].dot(self.weight.T)+self.bias)
        self.weight= self.weight + self.learning_rate*(self.train_data[i][-1]-temp)*self.train_data[i][0:-1]
        self.bias = self.bias + self.learning_rate*(self.train_data[i][-1]-temp)
        

        self.plot[0].remove()
        x = np.array(range(self.min_x, self.max_x+1))
        y = -(self.weight[0]*x + self.bias)/self.weight[1]
        self.plot=plt.plot(x, y ,linewidth = 1, color = 'black') #畫圖 ms：折點大小

    def update_3D(self, i): #3維資料更新參數
        temp=hardlim(self.train_data[i][0:-1].dot(self.weight.T)+self.bias)
        self.weight= self.weight + self.learning_rate*(self.train_data[i][-1]-temp)*self.train_data[i][0:-1]
        self.bias = self.bias + self.learning_rate*(self.train_data[i][-1]-temp)
        
        self.plot[0].remove()
        x = np.array(range(self.min_x, self.max_x+1))
        y = np.array(range(self.min_y, self.max_y+1))
        X, Y = np.meshgrid(x, y)
        Z = -(self.weight[0]*X + self.weight[1]*Y + self.bias)/self.weight[2]
        self.plot=[self.ax.plot_surface(X, Y, Z, color = 'black', alpha=.5)]#ax.plot_surface回傳的不是list需轉換

    def update_up_4D(self, i): #4維以上資料更新參數:
        temp=hardlim(self.train_data[i][0:-1].dot(self.weight.T)+self.bias)
        self.weight= self.weight + self.learning_rate*(self.train_data[i][-1]-temp)*self.train_data[i][0:-1]
        self.bias = self.bias + self.learning_rate*(self.train_data[i][-1]-temp)
        
    def gen_function_convergence1(self): # 禎數生成器 for epoch 的收斂條件
        for k in range(self.epoch):
            epoch_display.set(str(k))
            for i in range(self.train_count):      
                single_display.set(str(i+1))
                yield i
        epoch_display.set(self.epoch)
        single_display.set('0')
        
        #訓練完畢        
        btn1['state'] = tk.NORMAL
        btn2['state'] = tk.NORMAL
        btn3['state'] = tk.NORMAL
        data_combobox['state'] = tk.NORMAL
        weight_display.set("w1:  " + str('{:+.3f}'.format(self.weight[0]))+"\nw2: "+ str('{:+.3f}'.format(self.weight[1]))+"\nbias:  "+ str('{:+.3f}'.format(self.bias)))
        train_accuracy_display.set(str(self.evaluation(self.train_data, self.train_count)))
        test_accuracy_display.set(str(self.evaluation(self.test_data, self.test_count)))
        
    def gen_function_convergence2(self):  # 禎數生成器 for accuracy 的收斂條件
        count=0
        ans=0
        single_display.set(str(0))
        
        while(True):
            ans=self.evaluation(self.test_data, self.test_count)
            epoch_display.set(str(count))
            test_accuracy_display.set(str(ans))
            if(ans>=float(self.accuracy)):
                break #該epoch以達到期望正確率
                
            for i in range(self.train_count):
                single_display.set(str(i+1))
                yield i

            count+=1

        #訓練完畢        
        btn1['state'] = tk.NORMAL
        btn2['state'] = tk.NORMAL
        btn3['state'] = tk.NORMAL
        data_combobox['state'] = tk.NORMAL
        weight_display.set("w1:  " + str('{:+.3f}'.format(self.weight[0]))+"\nw2: "+ str('{:+.3f}'.format(self.weight[1]))+"\nbias:  "+ str('{:+.3f}'.format(self.bias)))
        train_accuracy_display.set(str(self.evaluation(self.train_data, self.train_count)))
        
    def check_learning_setting(self):
        if(data_combobox.current()==-1): 
            messagebox.showwarning("Warning","請選擇資料集")
            return False
        
        if(lr.get()==""):
            messagebox.showwarning("Warning","請輸入學習效率")
            return False
        else:
            try:
                self.learning_rate=float(lr.get())
                
            except:
                messagebox.showwarning("Warning","學習效率請輸入整數或浮點數")
                return False
                
        if(convergence.get()==0):
            if(epoch_accuracy.get()==""):
                messagebox.showwarning("Warning","請輸入Epoch")
            else:
                try:
                    self.epoch=int(epoch_accuracy.get())
                    return True
                except:
                    messagebox.showwarning("Warning","Epoch請輸入整數或浮點數")
                    return False
        else:
            if(epoch_accuracy.get()==""):
                messagebox.showwarning("Warning","請輸入Accuracy")
            else:
                try:
                    self.accuracy = float(epoch_accuracy.get())
                    return True
                except:
                    messagebox.showwarning("Warning","Accuracy請輸入整數或浮點數")
                    return False
                return
    def evaluation(self, data, count):
        correct=0
        for i in range(count):
            temp=hardlim(data[i][0:-1].dot(self.weight.T)+self.bias)
            if(temp==data[i][-1]):
                correct+=1

        return round(correct/count,3)
        
        
        
#---視窗
def creatobject(event):
    #event.widget.get()獲取選擇值
    pla = allfunction(event.widget.get())

#---激勵函數    
def hardlim(input): 
    if input>0:
        a=1
    else:
        a=0
    return a    

window = tk.Tk()
window.geometry("430x740")
window.resizable(False, False)
window.title("感知機訓練器")

file=["2Ccircle1", "2Circle1", "2CloseS", "2CloseS2", "2CloseS3", "2cring",\
      "2CS","2Hcircle1", "2ring", "5CloseS1", "C3D",\
      "perceptron1", "perceptron2", "perceptron3", "wine_2class"]

#設定框_1"
setting1 = tk.Frame(window)
setting1.grid(row=0, column=0, columnspan=2, sticky=tk.N, padx=15)

#設定框_1_學習率
lr = tk.StringVar()#學習率
lr.set("1")
tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="學習率：").grid(row=0, sticky=tk.W, pady=5)
tk.Entry(setting1, width=10, textvariable=lr).grid(row=1, sticky=tk.W)

#設定框_1_收斂條件
convergence = tk.IntVar() #判斷收斂條件
epoch_accuracy = tk.StringVar() #訓練次數、正確率
epoch_accuracy.set("2")
convergence_condition_text = tk.Label(setting1, font=("微軟正黑體", 12, "bold"), text="收斂條件：").grid(row=2, sticky=tk.W, pady=2)
tk.Radiobutton(setting1, font=("微軟正黑體", 10, "bold"), text="Epoch (浮點數自動捨去)", variable=convergence, value=0).grid(row=3, sticky=tk.W)
tk.Radiobutton(setting1, font=("微軟正黑體", 10, "bold"), text="Accuracy (測試資料)", variable=convergence, value=1).grid(row=4, sticky=tk.W)
tk.Entry(setting1, width=10, textvariable=epoch_accuracy).grid(row=5, sticky=tk.W)
#----------------------------------------------------------------

#設定框_2"
setting2 = tk.Frame(window)
setting2.grid(row=0, column=2, columnspan=2, sticky=tk.N, padx=15)

#設定框_2_視覺化更新速度：
tk.Label(setting2, font=("微軟正黑體", 12, "bold"), text="視覺化更新速度：").grid(row=0, sticky=tk.W, pady=4)
update_combobox = ttk.Combobox(setting2, value=['1倍速','2倍速','3倍速'], state="readonly") #readonly為只可讀狀態
update_combobox.grid(row=1, sticky=tk.W)
update_combobox.current(0) #預設Combobox為index0


#設定框_2_訓練資料集
tk.Label(setting2, font=("微軟正黑體", 12, "bold"), text="選擇訓練資料集").grid(row=2, sticky=tk.W, pady=4)
data_combobox = ttk.Combobox(setting2, value=file, state="readonly") #readonly為只可讀狀態
data_combobox.grid(row=3, sticky=tk.W)
data_combobox.bind("<<ComboboxSelected>>", creatobject)

#設定框_2_開始訓練按鈕
btn = tk.Button(setting2, text='開始訓練')
btn.grid(row=4, sticky=tk.E, pady=25)
#----------------------------------------------------------------
tk.Label(window, font=("微軟正黑體", 10, "bold"), text="開始訓練後，隨機將資料集中的 2/3 當作訓練資料；1/3 當做測試資料").grid(row=1, columnspan=4, pady=4)
btn1 = tk.Button(window, text='全部資料 (All)')
btn1.grid(row=2, column=0, columnspan=1, pady=4)
btn2 = tk.Button(window, text='訓練資料 (Train)')
btn2.grid(row=2, column=1, columnspan=2, pady=4)
btn3 = tk.Button(window, text='測試資料 (Test)')
btn3.grid(row=2, column=3, columnspan=1, pady=4)
#訓練圖
training_plot = tk.Frame(window, padx=5)
training_plot.grid(row=3, column=0,columnspan=4, padx=5)

fig = plt.figure(figsize=(4,4))

canvas = FigureCanvasTkAgg(fig, training_plot)  # A tk.DrawingArea.
canvas.get_tk_widget().pack(side=tk.TOP, expand=1)

#----------------------------------------------------------------
#訓練相關參數"

result_1 = tk.Frame(window)
result_1.grid(row=4, column=0, columnspan=2, sticky=tk.NW, padx=15)

lr_display = tk.StringVar()
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), text="學習效率： ").grid(row=0, column=0, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), textvariable=lr_display).grid(row=0, column=1, sticky=tk.W)

all_count_display = tk.StringVar()

train_count_display = tk.StringVar()
test_count_display = tk.StringVar()
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), text="全部資料數： ").grid(row=1, column=0, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), textvariable=all_count_display).grid(row=1, column=1, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), text="訓練資料數： ").grid(row=2, column=0, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), textvariable=train_count_display).grid(row=2, column=1, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), text="測試資料數： ").grid(row=3, column=0, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), textvariable=test_count_display).grid(row=3, column=1, sticky=tk.W)

weight_display = tk.StringVar()
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), text="鍵結值： ").grid(row=4, column=0, sticky=tk.W)
tk.Label(result_1, font=("微軟正黑體", 10, "bold"), textvariable=weight_display, justify=tk.LEFT).grid(row=5, column=0, columnspan=1, sticky=tk.W)


result_2 = tk.Frame(window)
result_2.grid(row=4, column=2, columnspan=2, sticky=tk.NW)
dataset = tk.StringVar()#資料集
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), text="資料集： ").grid(row=0, column=0, sticky=tk.W)
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=dataset).grid(row=0, column=1, columnspan=3, sticky=tk.W)
epoch_display = tk.StringVar()
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), text="Epoch： ").grid(row=1, column=0, sticky=tk.W)
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=epoch_display).grid(row=1, column=1, sticky=tk.W)
single_display = tk.StringVar()
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=single_display).grid(row=1, column=2, sticky=tk.W)
all_display = tk.StringVar()
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=all_display).grid(row=1, column=3, sticky=tk.W)
train_accuracy_display = tk.StringVar()
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), text="訓練資料正確率： ").grid(row=2, column=0, sticky=tk.W)
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=train_accuracy_display).grid(row=2, column=1, columnspan=3, sticky=tk.W)
test_accuracy_display = tk.StringVar()
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), text="測試資料正確率： ").grid(row=3, column=0, sticky=tk.W)
tk.Label(result_2, font=("微軟正黑體", 10, "bold"), textvariable=test_accuracy_display).grid(row=3, column=1, columnspan=3, sticky=tk.W)


tk.Label(result_2, font=("微軟正黑體", 10, "bold"), text="").grid(row=4, column=0, sticky=tk.W)


window.mainloop()

