# perceptron
## 程式簡介
### 使用說明
> 透過視覺化界面與動態更新展示「單層感知機( Perceptron )」的更新方法與使用
* 左方可以設定「學習率」；「收斂條件」：固定訓練次數 或 **測試資料**正確率

* 右方可以選擇「訓練資料集」；「視覺化更新速度」：感知機更新時動態顯示的速度

* 訓練時會將所選資料集分成2/3的訓練資料、1/3測試資料

* 下方可將所選資料集圖形化，支援2維、3維資料

* 訓練相關參數會顯示在最下方

* 訓練後必須**重新選擇**資料集來重制上次的訓練結果

 
### 範例圖
![](https://i.imgur.com/LNf6ZHh.png)

## Perceptron演算法
### 演算法簡介
> 一種用於「**分類**」的「**監督式學習**」演算法
* 又稱為Perceptron Learning Algorithm簡稱「PLA」

* Perceptron為所有AI算法最基礎的一種

* 資料是線性可分的形況下才能正確分類（演算法才會停止 )，所以必須設定停止條件

  ![](https://i.imgur.com/03YyVPq.png)

* 以生物神經元模型為基礎所開發，但只是跟生物神經元傳遞訊號的機制類似，除此之外沒有其他關係

* **神經網路架構圖：**

  ![](https://i.imgur.com/efcQkGr.png)
### 演算法步驟
#### 1. 前饋階段
* 假設一筆輸入向量 x 為 p 維度

* 將bias加入 x 中，即常數 1

* 感知機的權重 w 為 p+1 維度，即包含bias，隨機初始化

* 參數與激勵函數如下：

<img src="https://render.githubusercontent.com/render/math?math=x=[1, x_1, ..., x_p]"> ,  <img src="https://render.githubusercontent.com/render/math?math=w=[w_{bias}, w_1, ...,  w_p]">

<img src="https://render.githubusercontent.com/render/math?math=\phi(x)=\left\{\begin{array}{r} 1 \quad if \quad x > 0 \\ -1 \quad if \quad x < 0 \end{array} \right.">

感知機透過下式運作：

<img src="https://render.githubusercontent.com/render/math?math=v= 1*w_{bias} %2B (\sum_{i=1}^{p}x_i*w_{i}) ">

<img src="https://render.githubusercontent.com/render/math?math=output= \phi(v) ">


#### 2. 更新階段
前饋階段的輸出，1代表class 1；-1代表class 2。接下來要做更新的動作，如下：
>  x 表輸入向量、η表示學習率

<img src="https://render.githubusercontent.com/render/math?math=w(n%2b1)=\left\{\begin{array}{l} w(n)%2b\eta*x \quad if \quad x \in class1 \quad but \quad {w(n)}^T*x < 0 \\ w(n)%2d\eta*x \quad if \quad x \in class2 \quad but \quad {w(n)}^T*x > 0 \\w(n) \quad if \quad x \quad is \quad classified \quad correctly \end{array} \right." >

> 詳細收斂證明請參考台大林軒田老師的這部教學影片：https://reurl.cc/pmMb4d

#### 3. 疊代階段
持續做1、2步，直到達到停止( 收斂 )條件
