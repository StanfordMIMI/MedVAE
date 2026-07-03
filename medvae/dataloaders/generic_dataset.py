import os
from typing import Callable, List, Union

import polars as pl
from torch.utils.data import Dataset

__all__ = ["GenericDataset"]


class GenericDataset(Dataset):
    def __init__(
        self,
        split_path: Union[os.PathLike, str],
        data_dir: Union[os.PathLike, str],  # DEPRECATED
        dataset_id: int,
        *,
        split_column: str = None,
        split_name: str = None,
        img_column: str = None,
        img_dir: str = None,
        img_suffix: str = None,
        img_transform: Callable = None,
        lbl_columns: List[str] = None,
        lbl_transform: Callable = None,
        msk_column: str = None,
        msk_dir: str = None,
        msk_suffix: str = None,
        msk_transform: Callable = None,
        txt_column: str = None,
        txt_dir: str = None,
        txt_suffix: str = None,
        txt_transform: Callable = None,
        com_transform: Callable = None,
        **kwargs,
    ):
        """A Generic Dataset implementation.

        The generic dataset can be used to create any dataset from a given CSV "split" and a data
        directory. The dataset aims to be as flexible as possible, allowing the user to specify
        the columns to use for images, labels, and text. The user can also specify the transforms
        to apply to each of these components.

        Args:
            split_path (Union[os.PathLike, str]): Path to the split file.
            data_dir (Union[os.PathLike, str]): Path to the data directory.
            img_column (str, optional): Image column. Defaults to "image_uuid".
            img_suffix (str, optional): Image suffix. Defaults to ".npy".
            img_transform (Callable, optional): Image transform. Defaults to None.
            lbl_columns (List[str], optional): Label columns. Defaults to None.
            lbl_transform (Callable, optional): Label transform. Defaults to None.
            txt_column (str, optional): Text column. Defaults to None.
            txt_transform (Callable, optional): Text transform. Defaults to None.
            com_transform (Callable, optional): Composite transform. Defaults to None.
            dataset_id (int, optional): Dataset ID. Defaults to None.
        """

        # Store arguments
        self.split_path = split_path
        self.split_column = split_column
        self.split_name = split_name
        self.dataset_id = dataset_id
        self.img_dir = img_dir if img_dir is not None else data_dir
        self.img_column = img_column
        self.img_suffix = img_suffix
        self.img_transform = img_transform
        self.lbl_columns = lbl_columns
        self.lbl_transform = lbl_transform
        self.msk_column = msk_column
        self.msk_dir = msk_dir
        self.msk_suffix = msk_suffix
        self.msk_transform = msk_transform
        self.txt_column = txt_column
        self.txt_dir = txt_dir
        self.txt_suffix = txt_suffix
        self.txt_transform = txt_transform
        self.com_transform = com_transform
        self.kwargs = kwargs

        self.samples = {}

        # Create the samples for the images, labels, and text
        if not os.path.exists(split_path):
            raise ValueError(f"Split path {split_path} does not exist.")

        self.df = pl.read_csv(split_path)
        if isinstance(split_column, str) and isinstance(split_name, str):
            self.df = self.df.filter(pl.col(split_column) == split_name)

        # Generate image paths (list comprehension is polars-version agnostic;
        # Series.apply was renamed to map_elements in newer polars).
        if img_column is not None:
            self.samples["img"] = [
                os.path.join(self.img_dir, f"{x}{img_suffix or ''}")
                for x in self.df.get_column(img_column).to_list()
            ]
            
        # Extract the columns with labels
        if lbl_columns is not None:
            self.samples["lbl"] = self.df.select(lbl_columns)

        # Extract the column with text or a path to a text file
        if txt_column is not None:
            self.samples["txt"] = self.df.get_column(txt_column).to_list()

        # Extract the column with masks
        if msk_column is not None:
            self.samples["msk"] = [
                os.path.join(self.msk_dir, f"{x}{msk_suffix or ''}")
                for x in self.df.get_column(msk_column).to_list()
            ]

        self.print_stats()

    def __getitem__(self, idx: int):
        """Return a dictionary with the requested sample."""
        sample = {"group_id": self.dataset_id}

        # Image
        if "img" in self.samples:
            sample["img"] = self.samples["img"][idx]
            if callable(self.img_transform):
                sample["img"] = self.img_transform(sample["img"])

        # Labels
        if "lbl" in self.samples:
            sample["lbl"] = self.samples["lbl"][idx]
            if callable(self.lbl_transform):
                sample["lbl"] = self.lbl_transform(sample["lbl"])

        # Mask
        if "msk" in self.samples:
            sample["msk"] = self.samples["msk"][idx]
            if callable(self.msk_transform):
                sample["msk"] = self.msk_transform(sample["msk"])

        # Text
        if "txt" in self.samples:
            sample["txt"] = self.samples["txt"][idx]
            if callable(self.txt_transform):
                sample["txt"] = self.txt_transform(sample["txt"])

        # Common transform applied on sample level
        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self):
        return len(self.df)

    def print_stats(self):
        print(
            f"""
            === Dataset stats for split={self.split_name or "full"} ===
            CSV file: {self.split_path}
            Data directory: {self.img_dir}
            Number of samples: {len(self.df)}
        """
        )
        
    def get_labels(self):
        return self.samples["lbl"].to_numpy().flatten()
