#available losses  'L2freq', 'L1freq', 'L2time', 'L1time','L2mask', 'L1mask', 'SISDRtime', 'SISDRfreq','MinSNRsdsdr'

#python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss MinSNRsdsdr --model /home/enricguso/PycharmProjects/temp_experiments/e44_MinSNRsdsdr --epochs 460

#python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --model /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --softmask

python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/e46_CrossEntropy --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss CrossEntropy #--model /home/enricguso/PycharmProjects/temp_experiments/e45_BinaryCrossEntropy --epochs 1000
