# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia

#pip install -r requirements.txt

python aitw_data_gen.py --dataset 'general'
python aitw_data_gen.py --dataset 'single'
python aitw_data_gen.py --dataset 'install'
python aitw_data_gen.py --dataset 'web_shopping'
python aitw_data_gen.py --dataset 'google_apps'

