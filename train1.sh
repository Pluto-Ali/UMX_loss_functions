#'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',     'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy','LogL2', 'LogL1'


python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e47_logL1_time --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss LogL1 --model /home/enricguso/PycharmProjects/temp_experiments/e47_logL1_time --epochs 1000

#python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e47_logL1_time --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e47_logL1_time --model /home/enricguso/PycharmProjects/temp_experiments/e47_logL1_time --softmask
