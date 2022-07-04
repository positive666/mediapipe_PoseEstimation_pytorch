## MediaPipe human_poseEstimation in PyTorch

![Teaser](/doc/demo.gif)

For work reasons, open source can only go so far

based on https://github.com/zmurez/MediaPipePyTorch

![Teaser](/documentation/image/teaser.gif)


## feature

   1. Friendly Torch Implementation ,using the GPU the GPU
   
	1.1 Face detector (BlazeFace)
	
	1.2 Face landmarks
	
	1.3 Palm detector
	
	1.4 Hand landmarks
	
	1.5 Iris detector
	
   2. Add Iris recognition function
   3. Part of the interface for each body gesture is more flexible 
   4. 3d Avatar is given with reference to U3d



## driver model 
<p align="center">
    <a href="https://youtu.be/Jvzltozpbpk">
        <img src="doc/3d.gif">
    </a>
</p>

## install

  pip -r install requirements
  
## if your need 3D demo，provide a Unity DEMO
   
  (reference this repo:https://github.com/mmmmmm44/VTuber-Python-Unity)
  
## run 

  1. read args in demo.py ,such as：
  
    python demo.py --source sample\tesla.mp4 --detect_face  --detect_iris  --save_file 
  
  2. First,your need a 3D unity model and open project setting
  
    python demo.py --source sample\tesla.mp4 --detect_face   --connect --debug  --port 5066



## TODO
- [ ] imporve pose detect function
- [ ] Add conversion and verification scripts
- [ ] Add multiprocess demo
- [ ] verification performance 
- [ ] hand and body 3D model demo

## reference 

	1. https://github.com/zmurez/MediaPipePyTorch
	2. https://github.com/hollance/BlazeFace-PyTorch
	3. https://github.com/mmmmmm44/VTuber-Python-Unity