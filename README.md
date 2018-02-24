DSOD的Tensorflow实现。
-------------------

使用Tensorlayer和Skimage等，过程中使用了自己修改后的Tensorlayer增广，上传前修改应该能在原始TL跑，没测试代码。按论文可以在VOC07+12上训练吧，Loss那块Loc给了高权重。

源码框架参考https://github.com/lslcode/SSD_for_Tensorflow，做了一些修改。主要是Groundtruth那块还有增广。另外这个代码增广极度耗费CPU资源，训练效率很低，收敛也不快。有问题可以到https://zhuanlan.zhihu.com/p/33957333评论或者给我发邮件，colinyoo#outlook.com。虽然未必能给予帮助。


