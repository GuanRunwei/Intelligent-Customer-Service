import fasttext

def train_model():
    model = fasttext.train_unsupervised(input="wiki_cut_word.txt", model="skipgram", ws=6, minn=2, thread=12)
    model.save_model("fasttext.wiki.model.bin")


if __name__ == '__main__':
    train_model()
