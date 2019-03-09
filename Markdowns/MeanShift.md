# MeanShift算法

### 简介
Mean shift 中文译名又叫做，均值漂移。在机器学习领域可以实现聚类，在图像领域可以实现目标追踪和图像平滑。是一种应用很广的算法。

### 算法思想

#### 二维空间
![enter image description here](https://lh3.googleusercontent.com/BEDEiWALPtUjabAe5LOarkXLIQsOW8v8TxyEG40aGHCr6UA0i99tI6_Fl7hNnlQMKzWemr2mCq4S)
问题1：想象平面上有一些点，通过怎样的方法能找到点最密集的地方？
答：很简单，首先先随便选取一个点X，然后画一个半径为h的圆。之后求在这个圆中(圆形内部)的点到X所构成的向量的和。我们称之为平均向量。  
之后对X在平均向量上的方向和距离上进行平移。得到新的X。然后重复这个步骤。一直到它附近的点的平均向量为0，或者长度足够小。

问题2：依然是平面上有许多点，我们怎样能把这些点分成几类？（聚类）或者说有一个平面，该怎样对平面进行分割？
答：如果你是要分割平面，你要想象许多的点，把平面内的位置都覆盖了，你会发现最后所有的点也就只收敛于几种值。如果你要对已经有了的点进行聚类，那么你就只需要把要分类（实际是聚类）的点都进行一次计算就可以了。（这里有个小技巧，理论上在计算过程中所有经过的点，和包裹到的周围的点都可以认为是一个类别里的，这样在一些时候可以大大减少计算的量。）

#### 三维空间
![enter image description here](https://lh3.googleusercontent.com/-ZARHk3XHB2gOlUz5Yk1yz8cyEnZdXdHA24QBiR1qzExEfYuIBtWmh50hutxn3shDMjil3jLfX0c)
实际上和二维是一样的，只是把二维时候的圆换成三维的球。

### 推广到一般情况
在我们容易想象的二维和三维情况，这个问题很好解决，我们希望这个算法能适用于更广的范围。

#### 基本形式

在给定的  d  维的空间  $R^{d}$  中有  n  个样本点。这样不难得出在空间中任意一个点的Mean shift向量如下：

$$M _ { h } = \frac { 1 } { K } \sum _ { x _ { i } \in S _ { k } } \left( x _ { i } - x \right)$$

简单的解释一下这个公式。$K$ 是指的是 $x$ 有 $K$ 个临近点。其中超球体 $S_{k}$ 的半径为 $h$,它的公式表示如下:

$$S _ { h } = \left\{ y : \left( y - x _ { i } \right) _ { T } \left( y - x _ { i } \right) < h ^ { 2 } \right\}$$

#### 对平面上点的密度进行估计
我们想要找到平面上点最密集的地方，那么最简单的思路就是用一个函数表示出平面上每个点的密集程度，然后再找到这个函数的最大值就好了。
![enter image description here](https://lh3.googleusercontent.com/xVXXxdsqBRVVsgWCD8_fsu-NGu6-oDJ_xTcaM9nkZ-APlO-q__tzPteuQQ_FOUhsbHb1dJrb9psl)
我们要做的就是模拟出上图中左边的函数，再找到峰顶的位置。

我们先给出正确的密度公式：

$$f ( x ) = \frac { 1 } { n h ^ { d } } \sum _ { i = 1 } ^ { n } K \left( \frac { x - x _ { i } } { h } \right)$$

那上面这个密度公式是怎么来的呢？
我们先从一维的情况开始考虑。我们知道点在总个数为N。于是我们不妨选取一段很小的距离h，然后看看在这段范围内有多少的点。假设有k个点。于是他的密度可以近似的表示为：$p=kN$,想象一下这里面h足够的小，然后我们可以得到这样的图像。
![enter image description here](https://lh3.googleusercontent.com/fV5c_JPrzolrJrFC7pBLH6dxhQ9wJZk3QUVcbjEfNom-q37EHXFtecczxNjwetct1iSLr4Es9kmo)
然而 $\frac{k}{N}$ 仅仅是概率，并不是真正的密度。我们考虑物理中的密度计算为 $\frac{m}{v}$ ，我的理解是把概率看作物理中的质量，长度h看作v，所以就有密度为 $\frac{k}{Nh}$ 。在二维中就是 $\frac{k}{Nv}$，其中$v=h^{2}$ 为一个正方形窗口。
是现在表示整个密度的函数似乎太麻烦了。那怎么办呢？我们设想一个辅助函数。让他有下面三个功能：

1.  在窗口内有值，在窗口外没有值。
2.  这个函数表现出离x越近对密度的结果影响越大，越远越小。
3.  函数在各个方向上的影响是一致的。（对称的）

现在我们这样的函数起名为K，于是就可以轻易表示出来在二维空间中的密度函数。
$$ 
f(x)=\frac{1}{n h^{2}} \sum_{i=1}^{n} K\left(\frac{x-x_{i}}{h}\right)
 $$
 
 可以理解为单位密度 $\frac{1}{nh^{2}}$ 乘以所有点，经过K函数变换后值的合，就是这个窗口的密度。这个K函数，就是我们所说的 *核函数* 。而我们使用核函数的原因就是上面提到的核函数的三个功能。
  我们把现在得到的这个公式推广到高维，就得到了我们一开始给出的密度公式。
 
 下面再给出几个核函数的例子：
 ![enter image description here](https://lh3.googleusercontent.com/ld517NoNYoY0p0oN6dEKnFkcFoQ4z4NfzwP-AphFGty5b_1Xi8F1SzKQUaa8OBUTAtmNBkZkqAQs)

#### 找到密度最大的地方
现在我们已经有了平面上点的密度函数，那么下一步就是找到这个函数梯度为 0 ，也就是峰顶的位置，这就是我们要找的密度最大的地方。

首先要确定核函数的公式。
$$ 
K(x)=c_{k, d} k\left(\|x\|^{2}\right)
 $$
 将核函数代入到原来的密度估计公式，然后求梯度得到$\nabla f(x)$。之后令$\nabla f(x)=0$，
 $$ 
\nabla f(x)=\frac{2 c_{k, d}}{n h^{d+2}} \sum_{i=1}^{n}\left(x_{i}-x\right) g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)
 $$
 其中 $g(s)=-k^{\prime}(s)$ 。
 
 我们观察这个得到的式子，
 第一项为一个常数：
 $$ 
\frac{2 c_{k, d}}{n h^{d+2}}
 $$
 第二项：
 $$\sum_{i=1}^{n}\left(x_{i} \stackrel{e}{-} x\right)$$
 其实就是各点到点$x$的向量和。
 第三项：
 $$ 
g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)
 $$
这一项保证了在窗口h内有值，在窗口h外对梯度下降的影响是零。且在高斯核下，里 $x$ 越近的点在对 $(x_{i}-x)$ 这个向量的加权越大。
 
  高斯核函数：$$ 
k\left(x, x^{\prime}\right)=e^{-\frac{ \|x-x^{\prime} ]^{2}}{2 \sigma^{2}}}
 $$
令 $x'=0,k(x,0)$ 的图像如下
 ![enter image description here](https://lh3.googleusercontent.com/FDe6G1ejmS6k0jjkxF46GAGLDOwFPuKUE6Gh03jiXQUFpC3O8vGtZi-Fkx2BsjcinSOMU9mqUYiV)

```
可以看出x'和x的距离越远，值越小。
```
回顾一下我们现在有的公式：
$$ 
\nabla f(x)=\frac{2 c_{k, d}}{n h^{d+2}} \sum_{i=1}^{n}\left(x_{i}-x\right) g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)
 $$
 整理可得到：
 $$ 
\nabla f(x)=\frac{2 c_{k, d}}{n h_{d+2}}\left[\sum_{i=1}^{n} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)\right]\left[\frac{\sum_{i=1}^{n} x_{i} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)}{\sum_{i=1}^{n} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)}-x\right]
 $$
```
首先把x−xix−xi和后面的一大坨乘开。然后在每一项在除以那一大坨。在外面乘以那一大坨。就是这个公式了。
```
其中，$g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)$永远大于 0 ，且最后一项称之为meanshift向量，即：
$$ 
m_{h}(x)=\frac{\sum_{i=1}^{n} x_{i} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)}{\sum_{i=1}^{n} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)}-x
 $$
 ![enter image description here](https://lh3.googleusercontent.com/M9whig9Gp2OA77G9ETCnUyI3m12GNveRjAlm6u4aREGnrgYIHB3Mm9nvIRYFQrhhPR9leBV9WMEE)

从上述公式中可以发现，在x点的密度估计为：
$$ 
f_{G}(x)=\frac{C}{n h^{d}} \sum_{i=1}^{n} g\left(\left\|\frac{x-x_{i}}{h}\right\|^{2}\right)
 $$
 则密度估计梯度变为：
 $$ 
\hat{\nabla} f_{K}(x)=\hat{f}_{G}(x) \frac{2 / C}{h^{2}} M_{k, G}(x)
 $$
 那么我们可以得到meanshift向量：
 $$ 
M_{k, G(x)}=\frac{h^{2}}{2 / C} \frac{\nabla f_{K}(x)}{\hat{f}_{G}(x)}
 $$
上面这个式子表明：

 1. 在点x处，MeanShift向量于密度梯度仅差一个常量的比例系数。而梯度就是密度变化最大的方向，所有MeanShift向量总是指向密度增大最大的方向。
 2. MeanShift向量于点 x 的密度成反比，也就是说 x 点的密度越大，MeanShift向量的长度越小。也就是说这是一个变步长的自适应算法，可以避免出现在最优点附近振荡而无法收敛的情况。

反复进行以下步骤，就是MeanShift算法的过程：

 1. 计算MeanShift向量 $M_{k, G}(x)$；
 2. 根据 $M_{k, G}(x)$ 移动窗口；
 3. 直到 $M_{k, G}(x)$ 为0或者小于一个阈值；

