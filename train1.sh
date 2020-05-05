#'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq',     'MinSNRsdsdr', 'CrossEntropy', 'BinaryCrossEntropy','LogL2time', 'LogL1time', 'LogL2freq', 'LogL1freq'

python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e50_logL2_freq --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss LogL2freq  #--model /home/enricguso/PycharmProjects/temp_experiments/e50_logL2_freq --epochs 400

#python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e50_logL2_freq --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e50_logL2_freq --model /home/enricguso/PycharmProjects/temp_experiments/e50_logL2_freq --softmask
