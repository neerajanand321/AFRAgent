# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia

# pip install -r requirements.txt

# python aitw_data_gen.py --dataset 'general'
# python aitw_data_gen.py --dataset 'single'
# python aitw_data_gen.py --dataset 'install'
# python aitw_data_gen.py --dataset 'web_shopping'
# python aitw_data_gen.py --dataset 'google_apps'

accelerate launch --config_file="./configs/deepspeed_config.yaml" instructblip_main.py --data_root 'dataset' --bs 16 --all_data 1 --all_eval True --user_msg "all_data_any_res_adain_finetuning" --use_high_res True --train_any_res_adain True
