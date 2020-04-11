#available losses  'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq','MinSNRsdsdr'

#E36, FINAL BASELINE: JOINT MODEL WITH L2 ON MAGNITUDE ESTIMATES
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L2freq #--model /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --model /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --softmask

#E37,  JOINT MODEL WITH L1 ON MAGNITUDE ESTIMATES
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L1freq #--model /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --model /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --softmask

#E38, L2 ON IDEAL RATIO MASK
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --nb-workers 4 --hidden-size 1024 --lr 0.0001 --loss L2mask #--model /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --model /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --softmask

#E39, L1 ON IDEAL RATIO MASK
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L1mask #--model /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --model /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --softmask
