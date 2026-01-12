import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last=False):
        self.shuffle = shuffle
        self.dataset = dataset
        self.indices = np.arange(len(dataset), dtype=int)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples_drawn = 0

    def _shuffle_dataset(self):
        np.random.shuffle(self.indices)

    def __iter__(self):
        self.samples_drawn = 0
        if self.shuffle:
            self._shuffle_dataset()
        return self

    def __next__(self):
        if self.drop_last and self.samples_drawn + self.batch_size > len(self.dataset):
            raise StopIteration

        if self.samples_drawn >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[
            self.samples_drawn: min(
                len(self.dataset), self.samples_drawn + self.batch_size
            )
        ]

        self.samples_drawn += self.batch_size

        return [self.dataset[index] for index in batch_indices]
