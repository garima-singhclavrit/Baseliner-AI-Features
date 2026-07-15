source ~/anaconda3/etc/profile.d/conda.sh  # Adjust the path if using Anaconda or a different installation path
conda activate estimate 
sudo fuser -k 8000/tcp
sudo service mongod start
nohup python -u manage.py runserver 0.0.0.0:8000 > res.txt 2>err.txt &

