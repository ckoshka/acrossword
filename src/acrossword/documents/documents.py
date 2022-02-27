import asyncio
import glob
import gzip
import sys
from abc import ABC, abstractmethod
from io import BytesIO
from typing import (Any, Callable, Coroutine, Dict, Generator, List, Optional,
                    Tuple, Union)
from urllib import parse
import os
import aiohttp
import justext
import numpy as np
import orjson
import regex as re
import slate
from loguru import logger
from nltk.tokenize import sent_tokenize
from numpy import ndarray
from ..rankers.rank import Ranker
import itertools

def dump(data: Union[Dict, List], f: str) -> None:
    """
    Dumps a dictionary or list to a gzipped file.
    :param data: The data to dump.
    :param f: The file to dump to."""
    with gzip.open(f, "wb") as file:
        file.write(
            orjson.dumps(
                data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
            )
        )


def load(f: str) -> Dict:
    """
    Loads a dictionary from a gzipped file.
    :param f: The file to load from."""

    with gzip.open(f, "rb") as file:
        data = orjson.loads(file.read())
    return data


logger.add(
    sys.stdout,
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True,
    format="<blue>{file}</blue> <green>{function}</green> <yellow>{time}</yellow> <red>{level}</red> <cyan>{message}</cyan>",
)


def similarity(x: ndarray, y: ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.
    :param x: The first vector.
    :param y: The second vector."""
    return np.dot(x, y)


def dehyphenate(text):
    """
    Dehyphenates a string.
    :param text: The string to dehyphenate."""
    text = text.replace("-\n", " ").replace("\n", " ")
    text = re.sub("s+", " ", text)
    return text


def trim_short_sentences(text: list):
    """
    Trims short sentences from a list of sentences.
    """
    text = [t for t in text if len(t) > 120]
    return text


class Searchable(ABC):
    """
    An abstract class that defines the interface for a searchable object."""

    @abstractmethod
    async def search(self, query: str, top: int) -> List[str]:
        """
        Searches for a query in the object.
        :param query: The query to search for.
        :param top: The number of results to return.
        :return: A list of results."""

        pass


class Document(Searchable):
    """A Document just represents some text data made up of chunks, which are semantically indexed and searchable. 
    :param embedding_model: The embedding model to use for the Document.
    :param directory_to_dump: The directory to dump the Document to.
    :param title: The title of the Document.

    """

    def __init__(
        self, embedding_model: str, directory_to_dump: str, **kwargs: dict
    ) -> None:
        self.embedding_model = embedding_model
        self.directory_to_dump = directory_to_dump
        self.title: str
        self.chunks: dict
        self.__dict__.update(kwargs)

    async def extract_from_url(
        self, url: str, split_into_sentences: bool = False, chunk_size: int = 3
    ) -> None:

        self.title = url
        logger.debug(f"Downloading the url {url}")
        paras_joined = None
        async with aiohttp.request("GET", url) as resp:
            if ".pdf" not in url:
                html = await resp.text()
                logger.debug(f"Retrieved the html for the page\n{html[0:200]}")
                extracted = justext.justext(html, justext.get_stoplist("English"))
                logger.debug(
                    f"Extracted the text from the html\n{extracted[0].text[0:200]}"
                )
                logger.debug(f"Total length: {len(extracted)}")
                sentences = [p.text for p in extracted if not p.is_boilerplate]
                logger.debug(f"Total sentences: {len(sentences)}")
                if split_into_sentences:
                    paras_joined = ". ".join(sentences)
                    sentences = sent_tokenize(paras_joined)
            else:
                f = BytesIO(await resp.read())
                loop = asyncio.get_event_loop()
                doc = await loop.run_in_executor(None, slate.PDF, f)
                sentences = sent_tokenize(dehyphenate(doc.text(clean=False)))
                sentences = trim_short_sentences(sentences)
                split_into_sentences = True
                f.close()
        if len(sentences) == 0:
            raise Exception(f"No sentences found for the url: {url}")
        # Merge them into groups of chunk_size
        sentences = [
            "\n".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        logger.debug(f"Split the text into sentences\n{sentences[0:10]}")
        # final_length = round(len(sentences) * 0.75)
        # summarised_paragraphs = pythy.summarise_sentences(sentences, final_length)

        self.chunks = {p: [] for p in sentences}

    async def extract_from_file(
        self, filename: str, break_on_newline: bool = False
    ) -> None:

        with open(filename, "r") as file:
            text = file.read()
        if break_on_newline:
            sentences = text.split("\n")
        else:
            sentences = sent_tokenize(text)
        sentences = [s for s in sentences if len(s) > 20]
        self.chunks = {p: [] for p in sentences}
        self.title = filename

    async def embed(self) -> None:

        from torch import Tensor
        '''The import is inside this function because torch is a massive library and takes a prohibitive amount of time to load.'''

        ranker = Ranker()
        while await ranker.is_empty():
            await asyncio.sleep(0.2)
        logger.debug(f"Converting the chunks to embeddings")
        embeddings: List[Tensor] = await ranker.convert(
            model_name=self.embedding_model, sentences=tuple(self.chunks.keys())
        )
        logger.debug(f"Converted {len(self.chunks)} chunks to embeddings")
        self.chunks = {
            p: np.around(np.array(e), 6) for p, e in zip(self.chunks.keys(), embeddings)
        }
        embeddings_as_np_array: List[np.ndarray] = [e.numpy() for e in embeddings]
        logger.debug(f"Converted the embeddings to numpy array")
        self.embedding = np.mean(embeddings_as_np_array, axis=0)

    async def serialise(self) -> None:

        logger.debug(f"Serialising the document")

        loop = asyncio.get_event_loop()

        await loop.run_in_executor(None, dump, 
            self.__dict__,
            self.directory_to_dump + f"/{parse.quote_plus(self.title)}.json",
        )

    @classmethod
    async def from_url_or_file(
        cls,
        source: str,
        embedding_model: str,
        directory_to_dump: str,
        is_url: bool = False,
        is_file: bool = False,
        split_into_sentences: bool = False,
        split_on_newline: bool = False,
    ) -> "Document":
        document = cls(embedding_model, directory_to_dump)
        if parse.quote_plus(source) in os.listdir(directory_to_dump):
            await cls.deserialise(directory_to_dump + "/" + parse.quote_plus(source))
        if is_url:
            await document.extract_from_url(source, split_into_sentences)
        elif is_file:
            await document.extract_from_file(source, split_on_newline)
        else:
            raise ValueError("Please specify whether the source is a url or a file")
        await document.embed()
        await document.serialise()
        return document

    async def search(self, query: str, top: int, **kwargs: dict) -> List[str]:

        logger.debug(f"Searching for {query}")

        ranker = Ranker()

        query_embedding = await ranker.convert(
            model_name=self.embedding_model, sentences=tuple([query])
        )
        chunk_embeddings = [(p, np.array(e)) for p, e in self.chunks.items()]
        chunk_embeddings.sort(
            key=lambda x: similarity(x[1], np.array(query_embedding[0])), reverse=True
        )
        logger.debug(f"First chunk embedding looks like: {chunk_embeddings[0][1]}")
        # logger.debug(f"Top results were {chunk_embeddings[0:top]}")
        return [p for p, e in chunk_embeddings[:top]]

    @classmethod
    async def deserialise(cls, filepath: str) -> "Document":
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, load, filepath)
        return Document(**doc)


class DocumentCollection(Searchable):
    """Has similar methods such as query as Documents, but can contain multiple Documents.
    It implements search by first ranking Document centroids, then searching each Document"""

    def __init__(self, documents: List[Document], **kwargs: dict) -> None:
        self.documents: List[Document] = documents
        self.embedding: Optional[np.ndarray] = None
        self.__dict__.update(kwargs)

    def extend_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)

    def add_document(self, document: Document) -> None:
        self.documents.append(document)

    def calculate_new_centroid(self) -> None:
        """After a new document is added, this goes through the embeddings for all of its internal documents and uses np.mean to calculate the new centroid"""
        self.embedding = np.mean([d.embedding for d in self.documents], axis=0)

    async def search(self, query: str, top: int) -> List[str]:
        """Search for a query in the collection."""
        ranker = Ranker()
        document_embeddings = [(doc, doc.embedding) for doc in self.documents]
        query_embedding = await ranker.convert(
            model_name=ranker.default_model, sentences=tuple([query])
        )
        # Sort the document embeddings by their numpy dot-product with the query embedding
        document_embeddings.sort(
            key=lambda x: similarity(x[1], np.array(query_embedding[0])), reverse=True
        )
        logger.debug(f"Top result was {document_embeddings[0][0].title}")
        top_results_for_top_documents: tuple[List[str]] = await asyncio.gather(
            *[doc.search(query, top) for doc, emb in document_embeddings[:top*2]]
        )
        logger.debug(f"Top results were {top_results_for_top_documents}")
        joined_results = '\n'.join(list(itertools.chain.from_iterable(top_results_for_top_documents)))
        sentences = sent_tokenize(joined_results)
        ranked_results = await ranker.rank(
            texts=tuple(sentences),
            query=query,
            top_k=top,
            model=ranker.default_model,
        )
        return ranked_results[:top]

    @classmethod
    def deserialise(cls, directory: str) -> "DocumentCollection":

        documents: List[Document] = []
        for filepath in glob.glob(directory + "/*"):
            data = load(filepath)
            document = Document(**data)
            documents.append(document)
        return DocumentCollection(documents)

    @classmethod
    async def from_list(
        cls,
        kind: str,
        sources: List[str],
        directory_to_dump: str,
        model: str = "all-mpnet-base-v2",
        split_into_sentences: bool = True,
    ) -> "DocumentCollection":

        doc_collection = cls([])

        document_futures: list[Coroutine[Any, Any, Document]] = []
        if kind == "url":
            url = True
        else:
            url = False
        for s in sources:
            document_futures.append(
                Document.from_url_or_file(
                    s,
                    model,
                    directory_to_dump,
                    is_url=url,
                    is_file=not url,
                    split_into_sentences=split_into_sentences,
                )
            )
        documents: tuple[Document] = await asyncio.gather(*document_futures)
        for doc in documents:
            doc_collection.add_document(doc)
        return doc_collection
