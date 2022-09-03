import re, pickle, argparse, os, gen

process = lambda text: re.sub("[\s]+", " ", re.sub("[^a-zа-я0-9\s,.?!-]", "", text.lower()))

def main():
    parser = argparse.ArgumentParser(description="TMC.Train 1.0 Stable | Markov chains training for randomized text generation | Developer - tiom4eg (tiom4eg.t.me)")
    parser.add_argument("--input-dir", type=str, help="Path to dataset folder (or standart input, if path not specified). Only .txt files used for training.")
    parser.add_argument("--model", type=str, help="Path to generated model.", default="model.pkl")
    parser.add_argument("--width", type=int, help="Width of model's prefix.", default=2)
    args = parser.parse_args()
    dataset = []
    if not args.input_dir:
        dataset = process(input("Enter text on which model will be trained: "))
    else:
        for filename in os.listdir(args.input_dir):
            if filename.endswith(".txt"):
                try:
                    with open(f"{args.input_dir}/{filename}", "r", encoding="utf-8") as f:
                        dataset.append(process(f.read()))
                except UnicodeDecodeError:
                    print(f"Skipped {filename} because it raised UTF-8 error.")
    if not dataset:
        print("No data found in this dataset.")
        return
    model = gen.MarkovChain(args.width)
    model.fit(dataset)
    with open(args.model, "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
