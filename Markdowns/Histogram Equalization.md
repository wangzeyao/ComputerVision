# 直方图和直方图均衡化(Histogram Equalization)

### 图像直方(Image Histiograms)
灰度直方图是对每一阶灰度的像素数量进行计数。
像素数量常常可以改用像素的概率来表现： Pk=nk/N  其中k为灰阶，n为k灰阶像素的数量，N为像素数量总数

### 直方图均衡化(Histogram Equalization)

对于对比度不高的灰阶图，我们需要用一个*灰度映射函数 T* 来把输入的灰度值r映射为输出的灰度值s，即 **s=T( r )**
那么如何获得这个映射函数T呢？
我们先设任意灰度值t：
* 在原图像中的概率密度函数为$p_{f}$(t)
* 在输出的图像中的概率密度函数为$p_{g}$(t)

然后，我们可以得到相应的累计分布函数：
* $S_{f}$(n) = $\int_0^np_{f}(t)dt$
* $S_{g}$(n) = $\int_0^np_{g}(t)dt$

此时，为了保证：
* **在原图像中比灰度值r暗的像素，映射到新图像中仍然比s暗**
* **在原图像中比灰度值r亮的像素，映射后依然比s亮**

所以有：
$S_{f}( r )$ = $S_{g}[T( r )]$=$S_{g}(s)$   （1）

对其微分就得到：
$p_{f}(r) · ds$ · $p_{g}(s) · ds$    （2）

我们令变换$T(r)$ = $L · S_{f}( r )$ 那么：
$s=T(r)$
&nbsp;&nbsp;&nbsp;$= L · S_{f}( r )$
&nbsp;&nbsp;&nbsp;$= L ·\int_0^rp_{f}(t)d$

那么我们就能得到：
$\frac{ds}{dr}=L ·p_{f}(r)$  &nbsp;&nbsp; (3)
由（2）（3）得：
$p_{g}(s)=\frac{1}{L}$

我们可以看出映射后，图像g中各灰度为均匀分布，概率密度函数为常数1/L，各种灰度被均衡化了。
那么我们要找的映射函数: $T(r)=L ·\int_0^rp_f(t)$

![enter image description here](https://lh3.googleusercontent.com/IV5jbUqA-6VjQvSpisRCCT7N8OZywnVx80ZUOMrhe1AgN7UmBZXauTmzMm063yXZKPbk8k4pD7O3 "计算实例")






