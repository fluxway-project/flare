"""Explicit table loading fixtures for the public catalog provider contract."""

from contextlib import contextmanager


class MemoryCatalogProvider:
    def __init__(self, loader, *, identity="memory-test-catalog"):
        self.identity = identity
        self.loader = loader
        self.opens = 0
        self.closes = 0

    @contextmanager
    def open(self):
        self.opens += 1
        try:
            yield catalog_for_loader(self.loader)
        finally:
            self.closes += 1


def catalog_for_loader(loader):
    class Catalog:
        def load_table(self, identifier):
            return loader(identifier)

        def close(self):
            pass

    return Catalog()
