# NNDL_final
The final of Neural Network and Deep Learning
## NeRF
神经网络与深度学习 期末作业第3题 NeRF三维重建 <br>
### 数据集构建
具体操作见报告<br>
参考 https://github.com/Fyusion/LLFF 进行拍摄，并用COLMAP（ https://github.com/colmap/colmap/releases/tag/3.8 ）<br>
进行处理<br>
运行<pre><code>python imgs2poses.py ../data/nerf_llff_data/screen/</code></pre>进行格式转化<br>
### 建模
编辑好config文件，运行<pre><code>python run_nerf.py --config configs/screen.txt</code></pre><br>等待训练结束，在logs文件夹下输出一段视频即为最后的建模结果
## Transformers-cifar数据
运行训练<pre><code>python train.py -net simplevit -gpu -method cutmix</code></pre>
测试<pre><code>python train.py -net simplevit -gpu -method cutmix -resume</code></pre>
