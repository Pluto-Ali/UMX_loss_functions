#'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',     'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy','LogL2time', 'LogL1time', 'LogL2freq', 'LogL1freq', 'PSA'

python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e51_PSA --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss PSA  #--model /home/enricguso/PycharmProjects/temp_experiments/e51_PSA --epochs 1000

python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e51_PSA --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e51_PSA --model /home/enricguso/PycharmProjects/temp_experiments/e51_PSA --softmask
