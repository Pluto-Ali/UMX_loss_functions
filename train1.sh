#available losses  'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq','MinSNRsdsdr'

#python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/tidy --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss MinSNRsdsdr #--model /home/enricguso/PycharmProjects/temp_experiments/tidy --epochs 400

python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e42_sisdrfreq --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e42_sisdrfreq --model /home/enricguso/PycharmProjects/temp_experiments/e31_freqsisdr --softmask

#python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --nb-workers 4 --hidden-size 1024 --lr 1e-06 --loss L2time # --model /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --epochs 755

