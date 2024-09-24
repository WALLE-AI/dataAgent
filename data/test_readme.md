的较好结果有很大的提升。CTPN的计算效率是0.14s／image，它使用了very deep VGG16model。

###### 表3.7 ICDAR 2011,2013和2015上的最新结果


<table border="1" ><tr>
<td colspan="4" rowspan="1">ICDAR 2011</td>
<td colspan="5" rowspan="1">ICDAR 2013</td>
<td colspan="4" rowspan="1">ICDAR 2015</td>
</tr><tr>
<td colspan="1" rowspan="1">Method</td>
<td colspan="1" rowspan="1">P</td>
<td colspan="1" rowspan="1">R</td>
<td colspan="1" rowspan="1">F</td>
<td colspan="1" rowspan="1">Method</td>
<td colspan="1" rowspan="1">P</td>
<td colspan="1" rowspan="1">R</td>
<td colspan="1" rowspan="1">F</td>
<td colspan="1" rowspan="1">T(s)</td>
<td colspan="1" rowspan="1">Method</td>
<td colspan="1" rowspan="1">P</td>
<td colspan="1" rowspan="1">R</td>
<td colspan="1" rowspan="1">F</td>
</tr><tr>
<td colspan="1" rowspan="1">Huang[16]</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">0.75</td>
<td colspan="1" rowspan="1">0.73</td>
<td colspan="1" rowspan="1">YIn[16]</td>
<td colspan="1" rowspan="1">0.88</td>
<td colspan="1" rowspan="1">0.66</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.43</td>
<td colspan="1" rowspan="1">CNN Pro.</td>
<td colspan="1" rowspan="1">0.35</td>
<td colspan="1" rowspan="1">0.34</td>
<td colspan="1" rowspan="1">0.35</td>
</tr><tr>
<td colspan="1" rowspan="1">Yno[17]</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">0.66</td>
<td colspan="1" rowspan="1">0.73</td>
<td colspan="1" rowspan="1">Neumann[16]</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">0.72</td>
<td colspan="1" rowspan="1">0.77</td>
<td colspan="1" rowspan="1">0.40</td>
<td colspan="1" rowspan="1">Deep2Text</td>
<td colspan="1" rowspan="1">0.50</td>
<td colspan="1" rowspan="1">0.32</td>
<td colspan="1" rowspan="1">0.39</td>
</tr><tr>
<td colspan="1" rowspan="1">Huang[21]</td>
<td colspan="1" rowspan="1">0.88</td>
<td colspan="1" rowspan="1">0.71</td>
<td colspan="1" rowspan="1">0.78</td>
<td colspan="1" rowspan="1">Neumann[16]</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">0.71</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.40</td>
<td colspan="1" rowspan="1">HUST</td>
<td colspan="1" rowspan="1">0.44</td>
<td colspan="1" rowspan="1">0.38</td>
<td colspan="1" rowspan="1">0.41</td>
</tr><tr>
<td colspan="1" rowspan="1">Yim[21]</td>
<td colspan="1" rowspan="1">0.86</td>
<td colspan="1" rowspan="1">0.68</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">FASText [21]</td>
<td colspan="1" rowspan="1">0.84</td>
<td colspan="1" rowspan="1">0.69</td>
<td colspan="1" rowspan="1">0.77</td>
<td colspan="1" rowspan="1">0.15</td>
<td colspan="1" rowspan="1">AJOU</td>
<td colspan="1" rowspan="1">0.47</td>
<td colspan="1" rowspan="1">0.47</td>
<td colspan="1" rowspan="1">0.47</td>
</tr><tr>
<td colspan="1" rowspan="1">Zhang[17]</td>
<td colspan="1" rowspan="1">0.84</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.80</td>
<td colspan="1" rowspan="1">Zhang [10]</td>
<td colspan="1" rowspan="1">0.88</td>
<td colspan="1" rowspan="1">0.74</td>
<td colspan="1" rowspan="1">0.80</td>
<td colspan="1" rowspan="1">60.0</td>
<td colspan="1" rowspan="1">NJU-Text</td>
<td colspan="1" rowspan="1">0.70</td>
<td colspan="1" rowspan="1">0.36</td>
<td colspan="1" rowspan="1">0.47</td>
</tr><tr>
<td colspan="1" rowspan="1">TextFlow[21]</td>
<td colspan="1" rowspan="1">0.86</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.81</td>
<td colspan="1" rowspan="1">TextFlow[10]</td>
<td colspan="1" rowspan="1">0.85</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.80</td>
<td colspan="1" rowspan="1">0.94</td>
<td colspan="1" rowspan="1">StradVision1</td>
<td colspan="1" rowspan="1">0.53</td>
<td colspan="1" rowspan="1">0.46</td>
<td colspan="1" rowspan="1">0.50</td>
</tr><tr>
<td colspan="1" rowspan="1">Text-CNN[21]</td>
<td colspan="1" rowspan="1">0.91</td>
<td colspan="1" rowspan="1">0.74</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">Text-CNN[10]</td>
<td colspan="1" rowspan="1">0.93</td>
<td colspan="1" rowspan="1">0.73</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">4.6</td>
<td colspan="1" rowspan="1">StradVision2</td>
<td colspan="1" rowspan="1">0.77</td>
<td colspan="1" rowspan="1">0.37</td>
<td colspan="1" rowspan="1">0.50</td>
</tr><tr>
<td colspan="1" rowspan="1">Gupta[9]</td>
<td colspan="1" rowspan="1">0.92</td>
<td colspan="1" rowspan="1">0.75</td>
<td colspan="1" rowspan="1">0.82</td>
<td colspan="1" rowspan="1">Gupta [10]</td>
<td colspan="1" rowspan="1">0.92</td>
<td colspan="1" rowspan="1">0.76</td>
<td colspan="1" rowspan="1">0.83</td>
<td colspan="1" rowspan="1">0.07</td>
<td colspan="1" rowspan="1">Zhang</td>
<td colspan="1" rowspan="1">0.71</td>
<td colspan="1" rowspan="1">0.43</td>
<td colspan="1" rowspan="1">0.54</td>
</tr><tr>
<td colspan="1" rowspan="1">CTPN</td>
<td colspan="1" rowspan="1">0.89</td>
<td colspan="1" rowspan="1">0.79</td>
<td colspan="1" rowspan="1">0.84</td>
<td colspan="1" rowspan="1">CTPN</td>
<td colspan="1" rowspan="1">0.93</td>
<td colspan="1" rowspan="1">0.83</td>
<td colspan="1" rowspan="1">0.88</td>
<td colspan="1" rowspan="1">0.14°</td>
<td colspan="1" rowspan="1">CTPN</td>
<td colspan="1" rowspan="1">0.74</td>
<td colspan="1" rowspan="1">0.52</td>
<td colspan="1" rowspan="1">0.61</td>
</tr></table>

## FTSN模型

Dai，Huang 等人在2018年提出了FTSN（Fused Text Segmentation Networks）模型［7］，使用分割网络支持倾斜文本检测。它使用Resnet-101做基础网络，使用了多尺度融合的特征图。标注数据包括文本实例的像素掩码和边框，使用像素预测与边框检测多目标联合训练。

<!-- Feature extraction Feature fusion region proposing Text Instance prediction 4000 10 4008 801ppnt Ca 月 RCt RIN Ret 、 h x1+1)x2 Mam N0 Carsl, Xe ag1 T  Pea  - 282  A4.1004 Px+1x4 PSaOPoing Bsea An C4 24 θ Eng4 1034.2 C4 Aea Red  -->
![](https://textin-image-store-1303028177.cos.ap-shanghai.myqcloud.com/external/6086f555c6f71cc1)

###### 图3.8 FTSN检测模型

ICDAR 2015强力阅读竞赛的挑战4［17］。IC15包含1000次训练和500次测试Google 眼镜拍摄的附带图像，而不关注视点和图像质量。因此，文本比例，方向和分辨率的大的变化导致文本检测的困难。下表给出FTSN在ICDAR2015数据集上的性能指标。

###### 表3.8 FTSN在ICDAR2015数据集上的性能

