---
layout: post
comments: true
title:  "Openpilot, its model and driving in GTA"
excerpt: "During the quarantine 2020 I found myself playing GTA V and started working on getting openpilot to work on it"
---

If you want this in your real car, check [here](https://comma.ai/shop/products/comma-two-devkit).

## Introduction
First off let me give you an introduction about openpilot, openpilot is a open source self driving car software developed by [comma.ai](https://comma.ai/). Openpilot now supports 40 of the most popular cars in the world, inlcuding Toyotas, Hondas, Acuras and many more. Openpilot is a open source project on GitHub made for people to contribute, there are also several bounties for it, that you as a developer can claim in order to get a little reward for your work. It aims to be the Android for self driving cars and this is true already supporting as mentioned above many of the most popular cars in the world.

## The openpilot model
The openpilot model is being developed in house by comma ai but the end model files are open source and on github and are easily readable with Tensorflow. Two months back I attempted a minimal implementation of the model in pure python, all the way from predicting what the model does to parsing the output to displaying it to the user, the code for that can be found [here](https://github.com/littlemountainman/modeld).

### Input
The most basic inputs are, 2 images in the yuv420p, both 6 channels, then there is a desire one hot input with shape (1,8), say if you want to do a lane change you would give the model here a different combination and it would do a lane change to the right or to the left, this was of course also used in the model training, the beautiful thing here is that is all was learned from people driving. There is also a input called traffic convention which has an input shape of (0,2) which does of course make sense because it is a one hot vector so there are 2 possibilities so left and right traffic and a (1,512) state vector which tells the model about the state from the car. 

### The model 

In November of 2019, where I published my first blog post about behavioral cloning there was also a [talk](https://www.youtube.com/watch?v=oBklltKXtDE) from Andrej Karpathy, Tesla at Pytorch Devcon, where he explained the driving models and saying everything is based off of a ResNet50, openpilot used ResNet18 for quite a while but now they switched to Efficientnet-B2 for the openpilot 0.7.5 model. The outputs are the left and right lane, a path prediction using those the steering angle for the car gets implemented in controlsd in the openpilot code. Then there are longitudinal outputs for the control over the  brake and the gas. The model has the name supercombo because it also includes a pose net used for the lead car prediction and also for velocity estimation just from the images. The desire input and state input gets passed back to the model input from the output. 

<pre>
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_imgs (InputLayer)         [(None, 12, 128, 256 0                                            
__________________________________________________________________________________________________
permute (Permute)               (None, 128, 256, 12) 0           input_imgs[0][0]                 
__________________________________________________________________________________________________
efficientnet-b2 (Model)         (None, 4, 8, 1408)   6442016     permute[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 4, 8, 32)     45088       efficientnet-b2[1][0]            
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 4, 8, 32)     128         conv2d[0][0]                     
__________________________________________________________________________________________________
elu (ELU)                       (None, 4, 8, 32)     0           batch_normalization[0][0]        
__________________________________________________________________________________________________
desire (InputLayer)             [(None, 8)]          0                                            
__________________________________________________________________________________________________
traffic_convention (InputLayer) [(None, 2)]          0                                            
__________________________________________________________________________________________________
vision_features (Flatten)       (None, 1024)         0           elu[0][0]                        
__________________________________________________________________________________________________
snpe_desire_pleaser (Dense)     (None, 8)            72          desire[0][0]                     
__________________________________________________________________________________________________
snpe_traffic_pleaser (Dense)    (None, 2)            6           traffic_convention[0][0]         
__________________________________________________________________________________________________
proc_features (Concatenate)     (None, 1034)         0           vision_features[0][0]            
                                                                 snpe_desire_pleaser[0][0]        
                                                                 snpe_traffic_pleaser[0][0]       
__________________________________________________________________________________________________
pre_gru_dense (Dense)           (None, 1024)         1059840     proc_features[0][0]              
__________________________________________________________________________________________________
re_lu (ReLU)                    (None, 1024)         0           pre_gru_dense[0][0]              
__________________________________________________________________________________________________
rnn_state (InputLayer)          [(None, 512)]        0                                            
__________________________________________________________________________________________________
rnn_r (Dense)                   (None, 512)          524800      re_lu[0][0]                      
__________________________________________________________________________________________________
rnn_rr (Dense)                  (None, 512)          262656      rnn_state[0][0]                  
__________________________________________________________________________________________________
snpe_pleaser (Dense)            (None, 512)          262656      rnn_state[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 512)          0           rnn_r[0][0]                      
                                                                 rnn_rr[0][0]                     
__________________________________________________________________________________________________
rnn_z (Dense)                   (None, 512)          524800      re_lu[0][0]                      
__________________________________________________________________________________________________
rnn_rz (Dense)                  (None, 512)          262656      rnn_state[0][0]                  
__________________________________________________________________________________________________
rnn_rh (Dense)                  (None, 512)          262656      snpe_pleaser[0][0]               
__________________________________________________________________________________________________
activation (Activation)         (None, 512)          0           add_1[0][0]                      
__________________________________________________________________________________________________
add (Add)                       (None, 512)          0           rnn_z[0][0]                      
                                                                 rnn_rz[0][0]                     
__________________________________________________________________________________________________
rnn_h (Dense)                   (None, 512)          524800      re_lu[0][0]                      
__________________________________________________________________________________________________
multiply (Multiply)             (None, 512)          0           rnn_rh[0][0]                     
                                                                 activation[0][0]                 
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 512)          0           add[0][0]                        
__________________________________________________________________________________________________
add_2 (Add)                     (None, 512)          0           rnn_h[0][0]                      
                                                                 multiply[0][0]                   
__________________________________________________________________________________________________
one_minus (Dense)               (None, 512)          262656      activation_1[0][0]               
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 512)          0           add_2[0][0]                      
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 512)          0           activation_1[0][0]               
                                                                 snpe_pleaser[0][0]               
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 512)          0           one_minus[0][0]                  
                                                                 activation_2[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 512)          0           multiply_1[0][0]                 
                                                                 multiply_2[0][0]                 
__________________________________________________________________________________________________
dense_1_path (Dense)            (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_left_lane (Dense)       (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_right_lane (Dense)      (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_lead (Dense)            (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_long_x (Dense)          (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_long_v (Dense)          (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
dense_1_long_a (Dense)          (None, 256)          131328      add_3[0][0]                      
__________________________________________________________________________________________________
relu_1_path (ReLU)              (None, 256)          0           dense_1_path[0][0]               
__________________________________________________________________________________________________
relu_1_left_lane (ReLU)         (None, 256)          0           dense_1_left_lane[0][0]          
__________________________________________________________________________________________________
relu_1_right_lane (ReLU)        (None, 256)          0           dense_1_right_lane[0][0]         
__________________________________________________________________________________________________
relu_1_lead (ReLU)              (None, 256)          0           dense_1_lead[0][0]               
__________________________________________________________________________________________________
relu_1_long_x (ReLU)            (None, 256)          0           dense_1_long_x[0][0]             
__________________________________________________________________________________________________
relu_1_long_v (ReLU)            (None, 256)          0           dense_1_long_v[0][0]             
__________________________________________________________________________________________________
relu_1_long_a (ReLU)            (None, 256)          0           dense_1_long_a[0][0]             
__________________________________________________________________________________________________
dense_2_path (Dense)            (None, 256)          65792       relu_1_path[0][0]                
__________________________________________________________________________________________________
dense_2_left_lane (Dense)       (None, 256)          65792       relu_1_left_lane[0][0]           
__________________________________________________________________________________________________
dense_2_right_lane (Dense)      (None, 256)          65792       relu_1_right_lane[0][0]          
__________________________________________________________________________________________________
dense_2_lead (Dense)            (None, 256)          65792       relu_1_lead[0][0]                
__________________________________________________________________________________________________
dense_2_long_x (Dense)          (None, 256)          65792       relu_1_long_x[0][0]              
__________________________________________________________________________________________________
dense_2_long_v (Dense)          (None, 256)          65792       relu_1_long_v[0][0]              
__________________________________________________________________________________________________
dense_2_long_a (Dense)          (None, 256)          65792       relu_1_long_a[0][0]              
__________________________________________________________________________________________________
relu_2_path (ReLU)              (None, 256)          0           dense_2_path[0][0]               
__________________________________________________________________________________________________
relu_2_left_lane (ReLU)         (None, 256)          0           dense_2_left_lane[0][0]          
__________________________________________________________________________________________________
relu_2_right_lane (ReLU)        (None, 256)          0           dense_2_right_lane[0][0]         
__________________________________________________________________________________________________
relu_2_lead (ReLU)              (None, 256)          0           dense_2_lead[0][0]               
__________________________________________________________________________________________________
relu_2_long_x (ReLU)            (None, 256)          0           dense_2_long_x[0][0]             
__________________________________________________________________________________________________
relu_2_long_v (ReLU)            (None, 256)          0           dense_2_long_v[0][0]             
__________________________________________________________________________________________________
relu_2_long_a (ReLU)            (None, 256)          0           dense_2_long_a[0][0]             
__________________________________________________________________________________________________
meta_dense_1 (Dense)            (None, 256)          262400      vision_features[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 64)           65600       vision_features[0][0]            
__________________________________________________________________________________________________
dense_3_path (Dense)            (None, 256)          65792       relu_2_path[0][0]                
__________________________________________________________________________________________________
dense_3_left_lane (Dense)       (None, 256)          65792       relu_2_left_lane[0][0]           
__________________________________________________________________________________________________
dense_3_right_lane (Dense)      (None, 256)          65792       relu_2_right_lane[0][0]          
__________________________________________________________________________________________________
dense_3_lead (Dense)            (None, 256)          65792       relu_2_lead[0][0]                
__________________________________________________________________________________________________
dense_3_long_x (Dense)          (None, 256)          65792       relu_2_long_x[0][0]              
__________________________________________________________________________________________________
dense_3_long_v (Dense)          (None, 256)          65792       relu_2_long_v[0][0]              
__________________________________________________________________________________________________
dense_3_long_a (Dense)          (None, 256)          65792       relu_2_long_a[0][0]              
__________________________________________________________________________________________________
meta_relu_1 (ReLU)              (None, 256)          0           meta_dense_1[0][0]               
__________________________________________________________________________________________________
elu_1 (ELU)                     (None, 64)           0           dense[0][0]                      
__________________________________________________________________________________________________
relu_3_path (ReLU)              (None, 256)          0           dense_3_path[0][0]               
__________________________________________________________________________________________________
relu_3_left_lane (ReLU)         (None, 256)          0           dense_3_left_lane[0][0]          
__________________________________________________________________________________________________
relu_3_right_lane (ReLU)        (None, 256)          0           dense_3_right_lane[0][0]         
__________________________________________________________________________________________________
relu_3_lead (ReLU)              (None, 256)          0           dense_3_lead[0][0]               
__________________________________________________________________________________________________
relu_3_long_x (ReLU)            (None, 256)          0           dense_3_long_x[0][0]             
__________________________________________________________________________________________________
relu_3_long_v (ReLU)            (None, 256)          0           dense_3_long_v[0][0]             
__________________________________________________________________________________________________
relu_3_long_a (ReLU)            (None, 256)          0           dense_3_long_a[0][0]             
__________________________________________________________________________________________________
dense_1_desire_state (Dense)    (None, 128)          65664       add_3[0][0]                      
__________________________________________________________________________________________________
desire_final_dense (Dense)      (None, 32)           8224        meta_relu_1[0][0]                
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 32)           2080        elu_1[0][0]                      
__________________________________________________________________________________________________
dense_final_path (Dense)        (None, 128)          32896       relu_3_path[0][0]                
__________________________________________________________________________________________________
dense_final_left_lane (Dense)   (None, 128)          32896       relu_3_left_lane[0][0]           
__________________________________________________________________________________________________
dense_final_right_lane (Dense)  (None, 128)          32896       relu_3_right_lane[0][0]          
__________________________________________________________________________________________________
dense_final_lead (Dense)        (None, 128)          32896       relu_3_lead[0][0]                
__________________________________________________________________________________________________
dense_final_long_x (Dense)      (None, 128)          32896       relu_3_long_x[0][0]              
__________________________________________________________________________________________________
dense_final_long_v (Dense)      (None, 128)          32896       relu_3_long_v[0][0]              
__________________________________________________________________________________________________
dense_final_long_a (Dense)      (None, 128)          32896       relu_3_long_a[0][0]              
__________________________________________________________________________________________________
relu_1_desire_state (ReLU)      (None, 128)          0           dense_1_desire_state[0][0]       
__________________________________________________________________________________________________
desire_reshape (Reshape)        (None, 4, 8)         0           desire_final_dense[0][0]         
__________________________________________________________________________________________________
elu_2 (ELU)                     (None, 32)           0           dense_1[0][0]                    
__________________________________________________________________________________________________
relu_final_path (ReLU)          (None, 128)          0           dense_final_path[0][0]           
__________________________________________________________________________________________________
relu_final_left_lane (ReLU)     (None, 128)          0           dense_final_left_lane[0][0]      
__________________________________________________________________________________________________
relu_final_right_lane (ReLU)    (None, 128)          0           dense_final_right_lane[0][0]     
__________________________________________________________________________________________________
relu_final_lead (ReLU)          (None, 128)          0           dense_final_lead[0][0]           
__________________________________________________________________________________________________
relu_final_long_x (ReLU)        (None, 128)          0           dense_final_long_x[0][0]         
__________________________________________________________________________________________________
relu_final_long_v (ReLU)        (None, 128)          0           dense_final_long_v[0][0]         
__________________________________________________________________________________________________
relu_final_long_a (ReLU)        (None, 128)          0           dense_final_long_a[0][0]         
__________________________________________________________________________________________________
final_desire_state (Dense)      (None, 8)            1032        relu_1_desire_state[0][0]        
__________________________________________________________________________________________________
desire_pred (Softmax)           (None, 4, 8)         0           desire_reshape[0][0]             
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 12)           396         elu_2[0][0]                      
__________________________________________________________________________________________________
path (Dense)                    (None, 385)          49665       relu_final_path[0][0]            
__________________________________________________________________________________________________
left_lane (Dense)               (None, 386)          49794       relu_final_left_lane[0][0]       
__________________________________________________________________________________________________
right_lane (Dense)              (None, 386)          49794       relu_final_right_lane[0][0]      
__________________________________________________________________________________________________
lead (Dense)                    (None, 58)           7482        relu_final_lead[0][0]            
__________________________________________________________________________________________________
long_x (Dense)                  (None, 200)          25800       relu_final_long_x[0][0]          
__________________________________________________________________________________________________
long_v (Dense)                  (None, 200)          25800       relu_final_long_v[0][0]          
__________________________________________________________________________________________________
long_a (Dense)                  (None, 200)          25800       relu_final_long_a[0][0]          
__________________________________________________________________________________________________
desire_state (Softmax)          (None, 8)            0           final_desire_state[0][0]         
__________________________________________________________________________________________________
meta (Dense)                    (None, 4)            1028        meta_relu_1[0][0]                
__________________________________________________________________________________________________
flatten (Flatten)               (None, 32)           0           desire_pred[0][0]                
__________________________________________________________________________________________________
pose (Activation)               (None, 12)           0           dense_2[0][0]                    
==================================================================================================
Total params: 13,146,045
Trainable params: 13,078,413
Non-trainable params: 67,632
__________________________________________________________________________________________________
</pre>

What you can see above is a summary of the model output by tensorflow, now let's see what the lane and path output looks like, so here is a screenshot from my [openpilot minimal repository](https://github.com/littlemountainman/modeld).

<center>
	<img src="/assets/openpilotout.png" style="width:95%;height:95%;">
</center>

The image might seem odd to you now but let me explain it, the path and the lanes are being predicted for the next 192 meters the y- axis are the next 192 meters, if it doesn't make sense to you think about it and try to understand it. 


## GTA 

As we have now discussed some of the most important parts about the model, I can start showing you openpilot in GTA. To understand my progress on that I would like you to know how the messaging system betweet the openpilot model and the controls works. The messaging system consists of a Master and a subsriber, openpilot has a own system called msgq, in the past zmq has been used but zmq has the big issue of sending all the messages through the kernel, msgq solves this problem using a shared memory location. The msgq has multiple channels for example channels called model, sensorEvents where from one the model prediction get sent and to sensorEvents, sensor information get sent including imu and other sensors. 

So as you might see we have the first problem already, we don get any senor information out of GTA V, cars in GTA just don't have a can bus or an imu or anything like that that would make our life easier. So because controlsd wouldn't work without the sensors and would crash then, I first stared by making my own gta car interface which is essentially based off of a Honda civic from 2018. So rewriting that took me around 2 weeks, keep in mind I started with that project in the end of march. After those two quite frustrating weeks I had a working output for the steering angle and the gas and the brake. So using that I was able to make a simple input to GTA V and openpilot was essentially driving with the WASD keys with pynput but I wouldn't even consider this an average 13 year old GTA online player, so I wanted to have continuous input so I tried simulating a playstation controller, turns out I spent 5 days doing useless stuff trying to get the playstation controller to work, so I switched over to a xBox controller, so that I needed to figure out now is how do I convert the model ouput to actual controller input. Turns out you need to multiply the steering output with 2,5 and add 1600 to it. For the long model the control is just just a simple if statement passed to GTA. 

So what you need for acutally need is two PCs, one PC with windows and all the xbox drivers installed and one laptop or PC with Ubuntu 16.04 running openpilot with webcam, the video stream could also be send over the network or with some kind of NDI capturing system over the network but everything was already complicated enough so that I decided to stick with the webcam, I would recommend at least a 1080p webcam for any kind of openpilot webcam things, I used a Logitech C920 but the qualitiy still wasn't really outstanding. 

So to explain the whole flow all together: 

image -> ubuntu laptop -> predictions with the model -> converting all the long and lateral control output -> sending it over my local network with zmq to my gaming pc -> gaming pc is emulating the xbox controller inputs -> driving in GTA! 

So lets look at some final results:
<center>
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A virtual self driving car, <a href="https://twitter.com/comma_ai?ref_src=twsrc%5Etfw">@comma_ai</a> openpilot driving in GTA V. Add your favourite game ! <a href="https://t.co/f5IrbbA2cY">https://t.co/f5IrbbA2cY</a> <a href="https://t.co/VozZPzkOq9">pic.twitter.com/VozZPzkOq9</a></p>&mdash; littlemountainman (@littlemtman) <a href="https://twitter.com/littlemtman/status/1257393599211352070?ref_src=twsrc%5Etfw">May 4, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>
 Watch the video with sound, I explain something. As we all know a video tells more than a 1000 words, if you liked the post please consider adding my blog to your RSS feed or following me on twitter for more updates. 

Have a nice day !

Leon 





