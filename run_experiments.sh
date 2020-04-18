#available losses  'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq','MinSNRsdsdr', 'BCE_IRM', 'BCE_IBM'

#E36, FINAL BASELINE: JOINT MODEL WITH L2 ON MAGNITUDE ESTIMATES
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L2freq #--model /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --model /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --softmask

#E37,  L1 ON MAGNITUDE ESTIMATES
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L1freq #--model /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --model /home/enricguso/PycharmProjects/temp_experiments/e37_L1mag --softmask

#E38, L2 ON IDEAL RATIO MASK
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --nb-workers 4 --hidden-size 1024 --lr 0.0001 --loss L2mask #--model /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --model /home/enricguso/PycharmProjects/temp_experiments/e38_l2mask --softmask

#E39, L1 ON IDEAL RATIO MASK
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss L1mask #--model /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --model /home/enricguso/PycharmProjects/temp_experiments/e39_l1mask --softmask

#E40, L1 ON TIME DOMAIN
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e40_l1time --nb-workers 4 --hidden-size 1024 --lr 1e-05 --loss L1time #--model /home/enricguso/PycharmProjects/temp_experiments/e40_l1time --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e40_l1time --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e40_l1time --model /home/enricguso/PycharmProjects/temp_experiments/e40_l1time --softmask

#E41, L2 ON TIME DOMAIN
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --nb-workers 4 --hidden-size 1024 --lr 1e-05 --loss L2time #--model /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --model /home/enricguso/PycharmProjects/temp_experiments/e41_l2time --softmask

#E42, SISDR ON TIME DOMAIN
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e42_sisdr --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss SISDRtime #--model /home/enricguso/PycharmProjects/temp_experiments/e42_sisdr --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e42_sisdr --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e42_sisdr --model /home/enricguso/PycharmProjects/temp_experiments/e42_sisdr --softmask

#E43, SISDR ON MAGNITUDE ESTIMATES
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss SISDRfreq #--model /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --model /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --softmask

#E44, min(SNR, SDSDR) ON TIME DOMAIN
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss MinSNRsdsdr #--model /home/enricguso/PycharmProjects/temp_experiments/e43_sisdrfreq --epochs 1000 #uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --model /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --softmask

#E45, BINARY CROSS ENTROPY ON IDEAL RATIO MASKS
python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss BinaryCrossEntropy #--model /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --epochs 1000
#uncomment this for resuming training
python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --model /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --softmask

