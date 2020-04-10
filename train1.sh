#python train.py --root /home/enricguso/datasets/musdb18hq --is-wav --output /home/enricguso/PycharmProjects/temp_experiments/tidy --nb-workers 4 --hidden-size 1024 --lr 0.001 --loss MinSNRsdsdr #--model /home/enricguso/PycharmProjects/temp_experiments/tidy --epochs 400

python eval.py --no-cuda --root /home/enricguso/datasets/musdb18hq --outdir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --is-wav --evaldir /home/enricguso/PycharmProjects/temp_experiments/e36_finalbaseline --model /home/enricguso/PycharmProjects/temp_experiments/e22_final_baseline --softmask

