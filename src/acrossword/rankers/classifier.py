"""This module uses SentenceTransformers to produce a text classifier based on averaged sentence embeddings. It can be used to filter toxic outputs, or identify good ones."""

from ..rankers.rank import Ranker
from numpy import floating, mean
import numpy
from typing import (
    List,
    Union,
    Dict,
    Optional,
    Tuple,
    Coroutine,
    Callable,
    Generator,
    Iterable,
)
import asyncio
import pickle


class Category:
    ranker = Ranker()

    def __init__(self, name: str, centroid: numpy.ndarray):
        self.name = name
        self.centroid = centroid

    @classmethod
    async def from_sentences(
        cls, sentences: Iterable[str], name: str, model: Optional[str] = None
    ) -> "Category":
        if not model:
            model = cls.ranker.default_model
        sentence_embds = [numpy.array(t) for t in await cls.ranker.convert(
            model_name=model, sentences=list(sentences)
        )]
        centroid = mean(sentence_embds, axis=0)
        return Category(name=name, centroid=centroid)


class Classifier:
    ranker = Ranker()

    def __init__(self, categories: List[Category]):
        self.categories = categories

    @classmethod
    async def from_dict(
        cls, categories: Dict[str, List[str]], model: Optional[str] = None
    ) -> "Classifier":
        cat_list: List[Coroutine[None, None, Category]] = []
        for name, sentences in categories.items():
            cat_list.append(Category.from_sentences(sentences, name, model))
        category_list = await asyncio.gather(*cat_list)
        return Classifier(list(category_list))

    def add_category(self, category: Category) -> None:
        self.categories.append(category)

    async def create_category(
        self, name: str, sentences: List[str], model: Optional[str] = None
    ) -> None:
        self.add_category(await Category.from_sentences(sentences, name, model))

    async def classify(self, event: str, model: Optional[str] = None) -> str:
        if not model:
            model = self.ranker.default_model
        if isinstance(event, str):
            _event = tuple([event])
        embds = await self.ranker.convert(model_name=model, sentences=_event)
        distances: Dict[str, floating] = {
            category.name: numpy.linalg.norm(embds[0] - category.centroid)
            for category in self.categories
        }
        closest_category = min(distances.items(), key=lambda x: float(x[1]))[0]
        return closest_category

    @classmethod
    def from_pickle(cls, path: str) -> "Classifier":
        with open(path, "rb") as f:
            return pickle.load(f)
