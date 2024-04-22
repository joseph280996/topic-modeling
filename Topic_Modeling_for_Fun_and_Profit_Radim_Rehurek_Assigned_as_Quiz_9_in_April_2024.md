---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="wRbqovbSk2Kw" -->
# Topic Modeling for Fun and Profit

[Source](https://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html)

In this notebook we'll

* vectorize a streamed corpus
* run topic modeling on streamed vectors, using gensim
* explore how to choose, evaluate and tweak topic modeling parameters
* persist trained models to disk, for later re-use
* In the [previous notebook 1 - Streamed Corpora](https://radimrehurek.com/gensim_3.8.3/auto_examples/howtos/run_compare_lda.html) we used the 20newsgroups corpus to demonstrate data preprocessing and streaming.

Now we'll switch to the English Wikipedia and do some topic modeling. Link: https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="57ncQLTfjZ6V" outputId="df45eec7-d867-46b9-f277-cca0ed457430"
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("Begun at", now)
```

```python colab={"base_uri": "https://localhost:8080/"} id="RqamHsJEzxkI" outputId="3099143d-6dda-4d05-8866-fce2b205bd0d"
!pip install six cython numpy scipy ipython[notebook]
!pip install nltk gensim pattern requests textblob
!python -m textblob.download_corpora lite
!pip install --upgrade gensim
!pip install --upgrade smart_open
```

```python colab={"base_uri": "https://localhost:8080/"} id="WYJ7tKzj77MC" outputId="b4c1068d-91bb-4827-b466-f2c3d1b320a1"
!rm -f download_data.py && wget 'https://raw.githubusercontent.com/piskvorky/topic_modeling_tutorial/master/download_data.py'
#
# The older datasets are no longer available, use the latest one.
!sed -i 's/20140623/latest/g' download_data.py
#
# wikimedia sometimes refuses to connect due to excessive load
# use a mirror site instead. see https://dumps.wikimedia.org/mirrors.html
!sed -i 's|dumps.wikimedia.org|dumps.wikimedia.your.org|g' download_data.py
```

```python colab={"base_uri": "https://localhost:8080/"} id="BrjMhI9J9U3y" outputId="f05c4cbe-f7ef-42f5-9fb9-db66a38996fb"
!rm -rf ./data
!mkdir ./data
!python download_data.py ./data
```

```python id="wT2qidy64FPH"
# import and setup modules we'll be using in this notebook
import logging
import itertools

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))
```

```python id="uKw4VgXOhg7K"
# import and setup modules we'll be using in this notebook
import logging
import itertools

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))
```

```python colab={"base_uri": "https://localhost:8080/"} id="UnDbQQvxryxg" outputId="8ada7c0e-dda1-423e-875d-c9993fc1e50c"
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus
path_to_wiki_dump = datapath("enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2")
corpus_path = get_tmpfile("wiki-corpus.mm")
wiki = WikiCorpus(path_to_wiki_dump)  # create word->word_id mapping, ~8h on full wiki
MmCorpus.serialize(corpus_path, wiki)  # another 8h, creates a file in MatrixMarket format and mapping

texts = [' '.join(txt) for txt in wiki.get_texts()]
print(texts[0])
print(texts[1])
```

```python id="_jm3-BTuvXrk"
# import gensim.utils as utils
from smart_open import smart_open
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens
```

```python colab={"base_uri": "https://localhost:8080/"} id="Sd4hkTySwgqF" outputId="2edf5ee4-2d48-4fb8-dbe6-46fb76605355"
# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
wiki_file = './data/simplewiki-latest-pages-articles.xml.bz2'
stream = iter_wiki(wiki_file)
for title, tokens in itertools.islice(iter_wiki(wiki_file), 8):
    print (title, tokens[:10])  # print the article title and its first ten tokens
```

```python id="_t9ryoRPRWCN"
id2word = {0: u'word', 2: u'profit', 300: u'another_word'}
```

```python id="OqYSBQvaRbZO"
doc_stream = (tokens for _, tokens in iter_wiki(wiki_file))
```

```python colab={"base_uri": "https://localhost:8080/"} id="pZwEa-U4RnVU" outputId="85296d36-2d02-403d-f378-28af9cbde78d"
%time id2word_wiki = gensim.corpora.Dictionary(doc_stream)
print(id2word_wiki)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6_K37BtgR7D7" outputId="31cc7ae5-e0db-4e65-e1cd-4ce5976e2732"
# ignore words that appear in less than 20 documents or more than 10% documents
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
print(id2word_wiki)
```

```python colab={"base_uri": "https://localhost:8080/"} id="frannwbQlWlj" outputId="e41080d7-78db-4a90-a7a7-ebc825653410"
now = datetime.now()

print("Done with SimpleWiki at", now)
```

<!-- #region id="pEcd6HXKAgnM" -->

**Question 1:** Print all words and their ids from id2word_wiki where the word starts with "human".

**Note for advanced users:** In fully online scenarios, where the documents can only be streamed once (no repeating the stream), we can't exhaust the document stream just to build a dictionary. In this case we can map strings directly into their integer hash, using a hashing function such as MurmurHash or MD5. This is called the "[hashing trick](https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick)". A dictionary built this way is more difficult to debug, because there may be hash collisions: multiple words represented by a single id. See the documentation of [HashDictionary](https://radimrehurek.com/gensim/corpora/hashdictionary.html) for more details.
<!-- #endregion -->

<!-- #region id="UddfsG1OBc0g" -->
## Vectorization
A streamed corpus and a dictionary is all we need to create [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) vectors:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="JqeLJSt_-m-6" outputId="274eff49-bd87-4034-ee65-03ab5ce66919"
doc = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."
bow = id2word_wiki.doc2bow(tokenize(doc))
print(bow)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b5sHV6Jk-pVg" outputId="e6ee1adb-982c-4877-edad-7631b0d65663"
print(id2word_wiki[10882])
```

```python colab={"base_uri": "https://localhost:8080/"} id="9VcnibT9-t_C" outputId="d1810ec8-c448-4604-cfb3-319ef5589baa"
class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).

        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus(wiki_file, id2word_wiki)
vector = next(iter(wiki_corpus))
print(vector)  # print the first vector in the stream
```

```python colab={"base_uri": "https://localhost:8080/"} id="AHDqnZ5_wo6D" outputId="2298dd64-7713-4e38-c676-0bda6283b4f2"
len(vector)
max([pair[1] for pair in vector])

index = [pair[1] for pair in vector].index(15)
index
```

```python colab={"base_uri": "https://localhost:8080/"} id="w_m3SKAr_EXv" outputId="7d9bb13b-5811-433d-8aa8-d7a8270c76fd"
# what is the most common word in that first article?

(most_index, most_count) = max(vector, key=lambda pair: pair[1])
print(id2word_wiki[most_index], most_count)
```

```python colab={"base_uri": "https://localhost:8080/"} id="N3QnS-rh_JqG" outputId="61475a2f-fa4e-4598-80cb-4d92674c642a"
%time gensim.corpora.MmCorpus.serialize('./data/wiki_bow.mm', wiki_corpus)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_ClKm84f_P9q" outputId="581614bb-6241-465e-fd6a-b3d9d401a2c7"
mm_corpus = gensim.corpora.MmCorpus('./data/wiki_bow.mm')
print(mm_corpus)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R-idcBNh_Xq7" outputId="9ecad26c-7b82-4d1d-a523-2edc5b3e3ea2"
print(next(iter(mm_corpus)))
```

<!-- #region id="_ZKF5qXZB-Bl" -->
## Semantic transformations
Topic modeling in gensim is realized via transformations. A transformation is something that takes a corpus and spits out another corpus on output, using `corpus_out = transformation_object[corpus_in]` syntax. What exactly happens in between is determined by what kind of transformation we're using -- options are Latent Semantic Indexing (LSI), Latent Dirichlet Allocation (LDA), Random Projections (RP) etc.

Some transformations need to be initialized (=trained) before they can be used. For example, let's train an LDA transformation model, using our bag-of-words WikiCorpus as training data:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="njkzuglrTGqn" outputId="f6233081-4637-43a8-ebb2-362c730064c0"
from gensim.utils import SaveLoad
class ClippedCorpus(SaveLoad):
    def __init__(self, corpus, max_docs=None):
        """
        Return a corpus that is the "head" of input iterable `corpus`.

        Any documents after `max_docs` are ignored. This effectively limits the
        length of the returned corpus to <= `max_docs`. Set `max_docs=None` for
        "no limit", effectively wrapping the entire input corpus.

        """
        self.corpus = corpus
        self.max_docs = max_docs

    def __iter__(self):
        return itertools.islice(self.corpus, self.max_docs)

    def __len__(self):
        return min(self.max_docs, len(self.corpus))

clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)  # use fewer documents during training, LDA is slow
# ClippedCorpus new in gensim 0.10.1
# copy&paste it from https://github.com/piskvorky/gensim/blob/0.10.1/gensim/utils.py#L467 if necessary (or upgrade your gensim)
%time lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)
```

```python colab={"base_uri": "https://localhost:8080/"} id="xjfbz__dTRTN" outputId="6e663f93-61a4-4f95-f79b-b2f088777f9e"
_ = lda_model.print_topics(-1)  # print a few most important words for each LDA topic
```

```python colab={"base_uri": "https://localhost:8080/"} id="12WC9u78lvou" outputId="6c6ec660-ead6-47f2-e30a-088eee657f43"
now = datetime.now()

print("LDA Topic Models computed at", now)
```

<!-- #region id="Womr_J7-Ci1K" -->
More info on model parameters in [gensim docs](https://radimrehurek.com/gensim/models/lsimodel.html).

Transformation can be stacked. For example, here we'll train a TFIDF model, and then train Latent Semantic Analysis on top of TFIDF:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aB2lxricTWzp" outputId="785d7bc4-a966-4be2-8797-dde5bda60db1"
%time tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)
```

```python colab={"base_uri": "https://localhost:8080/"} id="6oYHwT0PTesl" outputId="7f485a1c-7811-4add-aba5-817321c70f9e"
%time lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)
```

<!-- #region id="d3qW0dJyC8dQ" -->

The LSI transformation goes from a space of high dimensionality (~TFIDF, tens of thousands) into a space of low dimensionality (a few hundreds; here 200). For this reason it can also seen as **dimensionality reduction**.

As always, the transformations are applied "lazily", so the resulting output corpus is streamed as well:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eI_d5YJfTqMt" outputId="9210e5c3-e7fb-4f4e-d4b5-ff19d20a74ae"
print(next(iter(lsi_model[tfidf_model[mm_corpus]])))
```

```python colab={"base_uri": "https://localhost:8080/"} id="6WZO-RasTzhD" outputId="9ddbfb45-9f1b-40ef-efe9-8b78be98b598"
# cache the transformed corpora to disk, for use in later notebooks
%time gensim.corpora.MmCorpus.serialize('./data/wiki_tfidf.mm', tfidf_model[mm_corpus])
%time gensim.corpora.MmCorpus.serialize('./data/wiki_lsa.mm', lsi_model[tfidf_model[mm_corpus]])
# gensim.corpora.MmCorpus.serialize('./data/wiki_lda.mm', lda_model[mm_corpus])
```

```python colab={"base_uri": "https://localhost:8080/"} id="KBCOELOET8C6" outputId="ee43a5c6-935b-4d50-86e2-eb6fcbb925a0"
tfidf_corpus = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm')
# `tfidf_corpus` is now exactly the same as `tfidf_model[wiki_corpus]`
print(tfidf_corpus)

lsi_corpus = gensim.corpora.MmCorpus('./data/wiki_lsa.mm')
# and `lsi_corpus` now equals `lsi_model[tfidf_model[wiki_corpus]]` = `lsi_model[tfidf_corpus]`
print(lsi_corpus)
```

```python colab={"base_uri": "https://localhost:8080/"} id="WY1w_tAHmVpu" outputId="01229c19-ccb3-446c-af1e-691a81098111"
now = datetime.now()

print("LSI Topic Models computed at", now)
```

<!-- #region id="E0Oebr8rDNzc" -->
## Transforming unseen documents
We can use the trained models to transform new, unseen documents into the semantic space:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="rI-uWLw2UCix" outputId="d8df394d-acfe-4b52-a8fa-098393c0a724"
text = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."

# transform text into the bag-of-words space
bow_vector = id2word_wiki.doc2bow(tokenize(text))
print([(id2word_wiki[id], count) for id, count in bow_vector])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Ct3v7t5uUKA3" outputId="209b441c-0b0b-4237-bad1-95f33f914944"
# transform into LDA space
lda_vector = lda_model[bow_vector]
print(lda_vector)
# print the document's single most prominent LDA topic
print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))
```

<!-- #region id="v1ZFOv5xDdiz" -->
**Question 2**: print text transformed into TFIDF space.

For stacked transformations, apply the same stack during transformation as was applied during training:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pUAn_JaXURuV" outputId="56b21e5b-6680-40ce-8230-bba3a8081288"
# transform into LSI space
lsi_vector = lsi_model[tfidf_model[bow_vector]]
print(lsi_vector)
# print the document's single most prominent LSI topic (not interpretable like LDA!)
print(lsi_model.print_topic(max(lsi_vector, key=lambda item: abs(item[1]))[0]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="-GwRtBZjUWKt" outputId="a3d3a268-c7af-4bfd-9ec5-422a543a7018"
# store all trained models to disk
lda_model.save('./data/lda_wiki.model')
lsi_model.save('./data/lsi_wiki.model')
tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save('./data/wiki.dictionary')
```

```python colab={"base_uri": "https://localhost:8080/"} id="y_9qJAv1UZ9z" outputId="f1683814-2081-4c2a-b510-5fcffa7011e2"

# load the same model back; the result is equal to `lda_model`
same_lda_model = gensim.models.LdaModel.load('./data/lda_wiki.model')
```

<!-- #region id="zj25epFWD4dv" -->
## Evaluation
Topic modeling is an **unsupervised task**; we do not know in advance what the topics ought to look like. This makes evaluation tricky: whereas in supervised learning (classification, regression) we simply compare predicted labels to expected labels, there are no "expected labels" in topic modeling.

Each topic modeling method (LSI, LDA...) its own way of measuring internal quality (perplexity, reconstruction error...). But these are an artifact of the particular approach taken (bayesian training, matrix factorization...), and mostly of academic interest. There's no way to compare such scores across different types of topic models, either. The best way to really evaluate quality of unsupervised tasks is to **evaluate how they improve the superordinate task, the one we're actually training them for**.

For example, when the ultimate goal is to retrieve semantically similar documents, we manually tag a set of similar documents and then see how well a given semantic model maps those similar documents together.

Such manual tagging can be resource intensive, so people hae been looking for clever ways to automate it. In [Reading tea leaves: How humans interpret topic models](http://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf), Wallach *et al* suggest a "word intrusion" method that works well for models where the topics are meant to be "human interpretable", such as LDA. For each trained topic, they take its first ten words, then substitute one of them with another, randomly chosen word (intruder!) and see whether a human can reliably tell which one it was. If so, the trained topic is **topically coherent** (good); if not, the topic has no discernible theme (bad):
<!-- #endregion -->

<!-- #region id="Ruh20UwcE-iu" -->
## Misplaced Words


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9hnOUHGUUgNh" outputId="2329d93a-e956-456c-d566-80c7f4d0e80c"
# select top 50 words for each of the 20 LDA topics
top_words = [[word for _, word in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
print(top_words)
```

```python colab={"base_uri": "https://localhost:8080/"} id="WBBVk3W-UnE-" outputId="10c9fad6-1055-4e6b-ac5c-fc78c18e083e"
# get all top 50 words in all 20 topics, as one large set
all_words = set(itertools.chain.from_iterable(top_words))

print("Can you spot the misplaced word in each topic?")

# for each topic, replace a word at a different index, to make it more interesting
replace_index = np.random.randint(0, 10, lda_model.num_topics)

replacements = []
for topicno, words in enumerate(top_words):
    other_words = all_words.difference(words)
    replacement = np.random.choice(list(other_words))
    replacements.append((words[replace_index[topicno]], replacement))
    words[replace_index[topicno]] = replacement
    print (topicno, ' '.join([str(w) for w in words[:10]]))
    # print("%i: %s" % (topicno, ' '.join(words[:10])))
```

```python colab={"base_uri": "https://localhost:8080/"} id="HLgEex1tUwK1" outputId="b8533aad-4838-4d3d-8aa4-824f3a2cfa82"
print("Actual replacements were:")
print(list(enumerate(replacements)))
```

```python id="bBlGBL5xU02d"
# evaluate on 1k documents **not** used in LDA training
doc_stream = (tokens for _, tokens in iter_wiki(wiki_file))  # generator
test_docs = list(itertools.islice(doc_stream, 8000, 9000))
```

```python id="H8we2ffzVPyh"
def intra_inter(model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    half = int(len(test_docs)/2)
    part1 = [model[id2word_wiki.doc2bow(tokens[: half])] for tokens in test_docs]
    part2 = [model[id2word_wiki.doc2bow(tokens[half :])] for tokens in test_docs]

    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))
```

```python colab={"base_uri": "https://localhost:8080/"} id="pUS3wj1mVXKS" outputId="5e909e87-b8aa-4478-d966-1f7f3b8df5e8"
print("LDA results:")
intra_inter(lda_model, test_docs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="5qIfitLMVkrg" outputId="bfd6dc1a-8be1-4f30-d65f-928d7febb561"
print("LSI results:")
intra_inter(lsi_model, test_docs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="2skCjSKyk1et" outputId="87a00928-dbc0-4864-ef64-35b75b6bc5cd"
now = datetime.now()

print("Ended at", now)
```
