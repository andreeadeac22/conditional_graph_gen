#LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH &&
#export LD_LIBRARY_PATH && 
source activate cenv && 
export LD_PRELOAD=/home/aid25/miniconda3/lib/libstdc++.so.6.0.25  &&
echo $LD_LIBRARY_PATH && 
echo $LD_PRELOAD && 
/sbin/ldconfig -p | grep stdc++ && 
export PYTHONPATH=/home/aid25/conditional_graph_gen/ &&
cd /home/aid25/conditional_graph_gen/fast_molvae/ && 
python cond_sample.py --testY ../data/zinc310k/frac0.2/ --testX ../data/zinc310k/frac0.2/test/  --vocab ../data/zinc310k/frac0.2/vocab.txt --model frac0.2_vae_model/model.iter-45000 --save_dir res_frac0.2_vae_model/  --data ../data/zinc310k/frac0.2/ 
