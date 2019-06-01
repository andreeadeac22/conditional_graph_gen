source activate cenv &&
export LD_PRELOAD=/home/aid25/miniconda3/lib/libstdc++.so.6.0.25  &&
echo $LD_LIBRARY_PATH &&
echo $LD_PRELOAD &&
/sbin/ldconfig -p | grep stdc++ &&
export PYTHONPATH=/home/aid25/conditional_graph_gen/ &&
cd /home/aid25/conditional_graph_gen/fast_molvae/ &&
python test_predi.py  --vocab ../data/zinc310k/frac0.5/vocab.txt --model vae_model7/model.iter-95000 --save_dir vae_model7
