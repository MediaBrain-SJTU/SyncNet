# SyncNet
[ECCV2022] Latency-Aware Collaborative Perception
<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="imgs/backbone.png"
         alt="无法显示图片时显示的文字"
         style="zoom:这里写图片的缩放百分比"/>
    <br>		<!--换行-->
    Backbone of our latency-aware collaborative perception system	<!--标题-->
    </center>
</div>

# Abstract
Collaborative perception has recently shown great potential to improve perception capabilities over single-agent perception. Existing collaborative perception methods usually consider an ideal communication environment. However, in practice, the communication system inevitably suffers from latency issues, causing potential performance degradation and high risks in safety-critical applications, such as autonomous driving. To mitigate the effect caused by the inevitable latency, from a machine learning perspective, we present the first latency-aware collaborative perception system, which actively adapts asynchronous perceptual features from multiple agents to the same time stamp, promoting the robustness and effectiveness of collaboration. To achieve such a feature-level synchronization, we propose a novel latency compensation mod- ule, called SyncNet, which leverages feature-attention symbiotic estimation and time modulation techniques. Experiments results show that the proposed latency aware collaborative perception system with SyncNet can outperforms the state-of-the-art collaborative perception method by 15.6% in the communication latency scenario and keep collaborative per-ception being superior to single agent perception under severe latency.

# SyncNet Architecture

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="imgs/dblstm.png"
         alt="无法显示图片时显示的文字"
         style="zoom:这里写图片的缩放百分比"/>
    <br>		<!--换行-->
    Backbone of our latency-aware collaborative perception system	<!--标题-->
    </center>
</div>
