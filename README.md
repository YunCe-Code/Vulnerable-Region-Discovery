# Vulnerable-Region-Discovery
If you want to discover vulnerable region for ImageNet:  
You should download imageNet validation set and put images in the images_val folder.  
Run command python3 attack.py --dataset ImageNet --pixels 1 --maxiter 100 --popsize 800  
If you want to discover vulnerable region for CIFAR-10 put the weight file for attacked DNNs into checkpoint folder.  
Run command python3 attack.py --dataset cifar10 --pixels 1 --maxiter 100 --popsize 200  
