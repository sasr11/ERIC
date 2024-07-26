cd 
python main.py --dataset AIDS700nef
nohup python main.py --dataset AIDS700nef > 1/demo2.log 2>&1 &
# 不输出日志
nohup python main.py --dataset AIDS700nef > /dev/null 2>&1 &


python main.py --dataset AIDS700nef --run_pretrain --pretrain_path model_saved/AIDS700nef/2024-07-19_11-07-53


nvidia-smi


top -p 
ps -p 3657834
 3657834
 3778024
 
