import os
import sys
sys.path.append('/home/inspur/yunce/one-pixel-attack-pytorch/models')
import numpy as np
import copy
import argparse


import torch
import torch.nn.functional as F

from utils import  group, get_testloader, get_model, setup_seed

import torch.nn.functional as F
from fsde import differential_evolution
import time 

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='chose from imagenet/cifar10')
parser.add_argument('--model', default='vgg16', help='target models')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=200, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=200, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--sr', type=float, default=4.0, help='sharing radius')
args = parser.parse_args()


def perturb_image(xs, img, mean, std):
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)
	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/5)
		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-mean[0])/std[0]
			imgs[count, 1, x_pos, y_pos] = (g/255.0-mean[1])/std[1]
			imgs[count, 2, x_pos, y_pos] = (b/255.0-mean[2])/std[2]
		count += 1
	return imgs

def extract_coordinate(parameters):
    num_pixels = int(parameters.shape[1]/ 5)
    coord_pos = np.zeros(shape=(parameters.shape[0], num_pixels, 2))
    for i in range(num_pixels):
        coord_pos[:, i, :] = parameters[:, [i*5, i*5+1]]
    return coord_pos

def remove_same(tuples, values):
    # Remove the individual with same coordinate and preserve the one cause the most damage.
    largest_tuples = {}
    for t, v in zip(tuples, values):
        key = (t[0], t[1])  # First two elements as the key
        if key not in largest_tuples or v > largest_tuples[key][1]:
            largest_tuples[key] = (t, v)
# Extract the tuples with the largest values
    result = [t for t, v in largest_tuples.values()]
    return result
      


def pairwise_euclidean(pos):
	'''
	compute the minimum euclidean distance between two set of point
	input: (popsize, pixels, (x, y))  Nxnx2
	'''
	if pos.shape[1] == 1:
		pos = np.reshape(pos, (pos.shape[0], 2))
		#one pixel attack just need to compute euclidean distance 
		dist = (pos[:, None, :] - pos[None,:,:])**2
		dist = (np.sum(dist, axis=-1, keepdims=False))**0.5
	else:
		dist = np.zeros(shape=(pos.shape[0], pos.shape[0]))
		for i in range(pos.shape[1]):
			pixel = pos[:,i,:]
			pixel = np.tile(pixel[:,None,:], (1, pos.shape[1], 1))
			pixel_dist = (pixel[:, None, :, :] - pos[None,:,:,:])**2
			pixel_dist = np.min(((np.sum(pixel_dist, axis=-1, keepdims=False))**0.5),axis=-1, keepdims=False)
			dist += pixel_dist
		dist = dist/pos.shape[1]
	return dist 
		
    	
		
def predict_classes(xs, img, target_class, net, mean, std, target=True):
	#change it to maximize the fitness 
	#Already verify
	with torch.no_grad():
		imgs_perturbed = perturb_image(xs, img.clone(), mean, std)
		input = imgs_perturbed.cuda()
		output = F.softmax(net(input),-1).data.cpu().numpy()
		predictions = copy.deepcopy(output[:, target_class])

	if not target:
		predictions = 1 - predictions
	return predictions 




def attack(batch_idx, img, label, net, target=None, pixels=1, maxiter=100, popsize=200, sharing_radius=4.0, image_size=224, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    # img: 1*3*W*H tensor
    # label: a number

    targeted_attack = target is not None
    target_calss = target if targeted_attack else label
    bounds = [(0, image_size), (0, image_size), (0,255), (0,255), (0,255)] * pixels
    popmul = max(1, popsize/len(bounds))
    predict_fn = lambda xs: predict_classes(
        xs, img, target_calss, net, mean, std, target is not None)
    callback_fn = None

    inits = np.zeros([int(popmul*len(bounds)), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*image_size
            init[i*5+1] = np.random.random()*image_size
            init[i*5+2] = np.random.normal(128,127)
            init[i*5+3] = np.random.normal(128,127)
            init[i*5+4] = np.random.normal(128,127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
                                            recombination=1.0, atol=-1, callback=callback_fn, 
                                            polish=False, init=inits, 
                                            sharing_radius=sharing_radius/image_size)
    xs = attack_result.x
    sucess_individual =[]
    attack_image = perturb_image(xs, img, mean, std)
    with torch.no_grad():
        attack_var = attack_image.cuda()
        predicted_probs = F.softmax(net(attack_var), -1)
        predicted_classes = torch.argmax(predicted_probs,-1).data.cpu().numpy()
        
    # selected the successful ones 
    if targeted_attack:
        sucess_index = np.where(predicted_classes==target_calss)[0]
    else:
        sucess_index = np.where(predicted_classes != label)[0]

    prob = predicted_probs
    if targeted_attack: 
        prob = prob[:, target]
    else: 
        prob = 1 -  prob[:, label]
    prob = prob.cpu().numpy()

    xs = xs.astype(int)
    # conf = np.take(prob.cpu().numpy(), sucess_index, axis=0)
    success_individual = np.take(xs, sucess_index, axis=0)
    success_prob = np.take(prob, sucess_index, axis=0)
    success_individual = success_individual.astype(int)
    # print (success_individual, success_prob)
    if len(success_individual) != 0 :
        success_individual = remove_same(success_individual, success_prob)
        return 1, success_individual
    return 0, [None]

	



def attack_all(net, loader, pixels=1, targeted=False, maxiter=100, popsize=200, sharing_radius=4.0, image_size=224, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    correct = 0
    success = 0
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            img_var = input.cuda()
            prior_probs = F.softmax(net(img_var),-1)
            value, indices = torch.max(prior_probs, 1)
        if target[0] != indices.data.cpu():
            continue
        
        correct += 1
        target = target.numpy()
        targets = [None] if not targeted else range(10) 
        for target_calss in targets:
            flag, x = attack(batch_idx, input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize,  sharing_radius=sharing_radius,image_size=image_size)
            success += flag
            if (targeted):
                success_rate = float(success)/(9*correct)
            else:
                success_rate = float(success)/correct
            if flag == 1:
                pixel_num = len(x)
                # sample['ind_output'] = ind_out 
                # sample['output'] = prior_probs.data.cpu().numpy()
                # best_conf = np.max(peaks_conf)*100
                # avg_conf = np.mean(peaks_conf)*100
                # if targeted:
                # 		logger_file.write('Idx:%d | Label:%d | target:%d | Success Rate:%.2f | Num:%d | Peaks:%d | Best Conf:%.1f |Avg Conf:%.1f'%(
                # 			batch_idx, target[0], target_calss, success_rate, len(x), len(peaks), best_conf, avg_conf
                # 		))
                # else:
                # 		logger_file.write('Idx:%d | Label:%d | target:%s | Success Rate:%.2f | Num:%d | Peaks:%d | Best Conf:%.1f |Avg Conf:%.1f'%(
                # 			batch_idx, target[0], 'None', success_rate, len(x), len(peaks), best_conf, avg_conf
                # 		))
                # logger_file.write('\n')
                print ('Idx:%d | Label:%d | target:%s | Success Rate:%.2f | Num:%d '%(
                			batch_idx, target[0], str(target_calss), success_rate, pixel_num
                ))
                # logger_file.flush()
        # if correct == args.samples:
        #     break
    return success_rate

def main():
    setup_seed(0)
    print ('Images Loading ......')
    testloader, image_size, mean, std = get_testloader(args.dataset)
    print ('Loading Attack Model ......')
    model = get_model(dataset=args.dataset, model=args.model )
    print ("==> Starting attack...")
    results = attack_all(model, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, sharing_radius=args.sr, image_size=image_size, mean=mean, std=std)
    print ("Final success rate: %.4f"%results)

if __name__ == '__main__':
	main()