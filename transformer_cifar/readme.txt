tensorboard --logdir='runs' --port=6006 --host='localhost'


python train.py -net resnet18 -gpu
python train.py -net resnet18 -gpu -resume

(上面是之前的 只需要跑下面的两句话就ok！！)

python train.py -net simplevit -gpu -method cutmix
python train.py -net simplevit -gpu -method cutmix -resume
