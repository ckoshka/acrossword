Acrossword is a small async wrapper around the SentenceBERT library. It has a convenient object-oriented API with two main purposes:

* **semantic search**
    * create miniature, powerful, cached semantic search engines from organised collections of documents
    * easily serialise and deserialise those documents in a gzipped JSON format
    * create documents from cleaned webpages and text files
    * search using different levels of granularity – from a book, to a chapter, to a single sentence

* **zero-shot text classification**
    * simply provide examples of each class, or something as simple as "This sentence is about X", and it will quite reliably classify it correctly

It's useful if you want to avoid larger bloated libraries with capabilities you don't need, and comes with zero fuss.

## Usage

```bash
pip install git+https://github.com/ckoshka/acrossword
```

### Documents

```python
from acrossword import Document, DocumentCollection

document_from_url = await Document.from_url_or_file(
    source="https://en.wikipedia.org/wiki/Semantic_search",
    embedding_model="all-mpnet-base-v2",
    is_url=True,
    directory_to_dump="your_directory"
)

document_from_file = await Document.from_url_or_file(
    source="your_file.txt",
    embedding_model="all-mpnet-base-v2",
    is_file=True,
    directory_to_dump="your_directory"
)

await document_from_file.serialise()

document_from_file = await Document.deserialise(
    "your_directory/your_file.txt.json"
)

collection = DocumentCollection(documents=[
    document_from_url, 
    document_from_file, 
])

await collection.search("a sci-fi book with a plot about a librarian", top=3)

# You can also nest documentcollections within each other to create a hierarchy of documents
```

### Ranker

```python
from acrossword import Ranker
ranker = Ranker() #loads mpnet model by default, but you can specify anything from huggingface and local models you have already downloaded

embeddings = await ranker.convert(
    model_name=ranker.default_model, 
    sentences=["What's the capital of Paris?", "Didn't you mean France?"]
)

top_results = await ranker.rank(
    texts=["Mercury", "Uranus", "Pluto", "the Sun", "Earth", "Mars"],
    query="A celestial object known for being very hot",
    top_k=2,
    model=ranker.default_model
)

top_results = await ranker.weighted_rank(
    texts=["Mercury", "Uranus", "Pluto", "the Sun", "Earth", "Mars"],
    queries=["A celestial object known for being very hot", "A Roman god associated with messengers"],
    weights=[0.5, 0.5],
    top_k=2,
    model=ranker.default_model
)
```

### Classifier

```python
from acrossword import Classifier, Category

negative = await Category.from_sentences(["This is a negative sentence"], name="negative")
positive = await Category.from_sentences(["This is a positive sentence"], name="positive")

classifier = Classifier([positive, negative])

sentence1 = "My dog caught on fire then I lost my house in a hurricane"
sentence2 = "I won the lottery and got sent an infinite supply of raspberry jam"

await classifier.classify(sentence1)
# returns "negative"

```

### Server

```python
from acrossword import run
run()
```

This is helpful if you're doing lots of tests where you need to reload the model every time. It runs a local server and if a Ranker object detects that one is running when it's initialised, it will redirect all API calls to the server.