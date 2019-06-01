#LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH &&
#export LD_LIBRARY_PATH && 
source activate cenv && 
export LD_PRELOAD=/home/aid25/miniconda3/lib/libstdc++.so.6.0.25  &&
echo $LD_LIBRARY_PATH && 
echo $LD_PRELOAD && 
/sbin/ldconfig -p | grep stdc++ && 
export PYTHONPATH=/home/aid25/conditional_graph_gen/ &&
cd /home/aid25/conditional_graph_gen/fast_molvae/ && 
python vae_train.py --train zinc310k-processed/frac0.5/ --data ../data/zinc310k/frac0.5/ --save_dir vae_model8/ --infomax_factor_true 200 --infomax_factor_false 100 --load_epoch 170000 --beta 0.130
