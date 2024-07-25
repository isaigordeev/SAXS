#source venv38/bin/activate

python data_generation/generation.py -p "cubic" --mp "Pn3m"
python data_generation/processing.py
python data_generation/main.py -p "cubic" --mp "Pn3m"