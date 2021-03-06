#'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',     'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy','LogL2time', 'LogL1time', 'LogL2freq', 'LogL1freq', 'PSA', 'SNRPSA', 'Dissimilarity'

python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e53_Dissimilarity --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss Dissimilarity  #--model /home/enricguso/PycharmProjects/temp_experiments/e52_SNRPSA --epochs 300

#python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e52_SNRPSA --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e52_SNRPSA --model /home/enricguso/PycharmProjects/temp_experiments/e52_SNRPSA --softmask
