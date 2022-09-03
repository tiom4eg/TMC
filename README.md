# TMC
Randomixed text generation model based on Markov chains.

# Structure
`train.py` - trains new or existing model on some dataset, consisting of UTF-8 friendly .txt files

`gen.py` - uses already trained models to generate randomized text of specified length from scratch or using user-provided text.

# Usage
Launch `python train.py -h` or `python gen.py -h` in shell to see all parameters.

# Examples
`python train.py --input-dir texts-ru --model ru-big.pkl --width 3`: trains new 3-gram model with .txt files from texts-ru/ and saves it in ru-big.pkl.

`python gen.py --model eng-big.pkl --prefix "hey vsauce michael here" --length 1337`: generates randomized text of length 1337 using model saved as eng-big.pkl and user-provided text.
