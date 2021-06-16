python main.py --model 'ConvFeature(width=3,hidden=1024)' --dataset CIFAR10 --predictor-hidden-size 512 --loss 'cross_entropy' --p-start 8 --p-end 1000 --epochs 0,100,100,750,800 --kappa 1.0 --eps-test 0.03137 --eps-train 0.03451 -b 512 --lr 0.02 --wd 5e-3 --visualize -p 200 --swa