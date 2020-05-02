#'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',     'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy','LogL2', 'LogL1'


#python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss BinaryCrossEntropy --model /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --epochs 1000

#python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --model /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --softmask

python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e48_logL2_time --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss LogL2  --model /home/enricguso/PycharmProjects/temp_experiments/e48_logL2_time --epochs 550

python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e48_logL2_time --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e48_logL2_time --model /home/enricguso/PycharmProjects/temp_experiments/e48_logL2_time --softmask
