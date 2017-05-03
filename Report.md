## Very Deep Convolutional Networks ##
### 研究目的 ###

探究随着层数的增加，卷积网络的效果的发生什么变化

### 网络架构 ###

####图像预处理####
- 输入图像大小为224×224

####卷积层####

- 卷积步长为1
- 3×3的过滤器，其中有一个配置使用了1×1的过滤器

####Max-pooling层####
- 5层
- 2×2窗口，步长为2

####全连接层####

- 前2层每个有4096通道
- 最后1层有1000通道

All hidden layers are equiped with rectification non-linearity.

####配置（Network D 16 weight layers）####
- 13 卷积层 + 3 全连接层
- 卷积层宽度包括64, 128, 256, 512
- 参数数量：138 million
- 3×3 filter width

###分类的框架###

####训练####
- 图片大小：
  裁剪成224×224的大小
- Network weight初始化：
  用Network A的层来初始化其他网络的前4个卷积层和后3个全连接层


####测试####
- 第1个全连接层转化为7×7的卷积层,最后2个全连接层转化为1×1的卷积层

###分类的实验###

####分类效果的评估####
用top1-error和top5-error评估

- Singal Scale Evalution:
经过验证表明，更深次的网络、更小的filter效果较好
- Multi-Scale Evalution:
- Multi-Crop Evalution:
- Convent Fushion:

###结论###

卷积网络层数的增加可以提高分类的准确度