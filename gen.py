import re, pickle, argparse, random, collections, numpy as np

class MarkovChain:
    def __init__(self, width):
        self.width, self.words, self.trans = width, [], dict()

    def fit(self, dataset):
        for text in dataset:
            window, text = collections.deque(), text.split()
            self.words += text[:-1]
            for word in text:
                popped = collections.deque()
                for i in range(min(len(window), self.width), 0, -1):
                    phrase = ' '.join(window)
                    if not self.trans.get(phrase):
                        self.trans[phrase] = dict()
                    if not self.trans[phrase].get(word):
                        self.trans[phrase][word] = 0
                    self.trans[phrase][word] += 1
                    popped.append(window.popleft())
                window = popped
                if (len(window) == self.width):
                    window.popleft()
                window.append(word)

    def generate(self, prefix, length):
        result = prefix.split()
        window = collections.deque()
        window.extend(result[-min(len(result), self.width):])
        for i in range(length):
            while True:
                if not window:
                    window.append(random.choice(self.words))
                phrase = ' '.join(window)
                if self.trans.get(phrase):
                    counts = np.array(list(self.trans[phrase].values()))
                    word = np.random.choice(list(self.trans[phrase].keys()), p = counts / counts.sum())
                    if (len(window) == self.width):
                        window.popleft()
                    window.append(word)
                    result.append(word)
                    break
                else:
                    window.popleft()
        return result

process = lambda text: re.sub("[\s]+", " ", re.sub("[^a-zа-я0-9\s,.?!-]", "", text.lower()))

def main():
    parser = argparse.ArgumentParser(description="TMC.Gen 1.0 Stable | Generation of randomized texts using trained Markov chains | Developer - tiom4eg (tiom4eg.t.me)")
    parser.add_argument("--model", type=str, help="Path to the file from which the model is loaded.", required=True)
    parser.add_argument("--prefix", type=str, help="Beginning of the generated text.", default='')
    parser.add_argument("--length", type=int, help="Length of the generated text (in words).", default=0)
    args = parser.parse_args()
    try:
        with open(args.model, "rb") as f:
            model = pickle.load(f)
    except Exception:
        print("An error occurred while loading the model. Check the correctness of the specified path.")
    print(f"Generated text: ")
    print(' '.join(model.generate(process(args.prefix), args.length)))
    
if __name__ == '__main__':
    main()
