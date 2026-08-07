"""TheVault-specific data loading helpers for multi-label DPR retrieval."""

import collections
import logging
from typing import List

import jsonlines
from omegaconf import DictConfig

from dpr.data.biencoder_data import BiEncoderSample, get_dpr_files
from dpr.data.retriever_data import QASample, QASrc
from dpr.utils.data_utils import Dataset

logger = logging.getLogger(__name__)

TheVaultPassage = collections.namedtuple("TheVaultPassage", ["text", "title", "id"])


class TheVaultMultiLabelDataset(Dataset):
    """Loads prepared TheVault JSONL while retaining document IDs."""

    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
    ):
        super().__init__(
            selector,
            special_token=special_token,
            encoder_type=encoder_type,
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix,
        )
        self.file = file
        self.data_files = get_dpr_files(file)

    def calc_total_data_len(self):
        if not self.data:
            self.load_data()
        return len(self.data)

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if self.data:
            return
        records = []
        for path in self.data_files:
            with jsonlines.open(path, mode="r") as reader:
                records.extend(reader)
        records = [record for record in records if record.get("positive_ctxs")]
        if start_pos >= 0 and end_pos >= 0:
            records = records[start_pos:end_pos]
        self.data = records
        logger.info("Loaded %d TheVault multi-label samples", len(self.data))

    def __getitem__(self, index) -> BiEncoderSample:
        record = self.data[index]
        sample = BiEncoderSample()
        sample.query = self._process_query(record["question"])
        sample.query_id = str(record.get("query_id", index))
        sample.positive_doc_ids = [str(value) for value in record.get("positive_ids", [])]

        def passages(key: str) -> List[TheVaultPassage]:
            result = []
            for passage in record.get(key, []):
                result.append(
                    TheVaultPassage(
                        text=passage["text"],
                        title=passage.get("title"),
                        id=str(passage["id"]),
                    )
                )
            return result

        sample.positive_passages = passages("positive_ctxs")
        sample.negative_passages = passages("negative_ctxs")
        sample.hard_negative_passages = passages("hard_negative_ctxs")
        if not sample.positive_doc_ids:
            sample.positive_doc_ids = [passage.id for passage in sample.positive_passages]
        return sample


class TheVaultMultiLabelQASrc(QASrc):
    """Evaluation query source whose labels are relevant document IDs."""

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as reader:
            for record in reader:
                data.append(
                    QASample(
                        self._process_question(record["question"]),
                        str(record.get("query_id", len(data))),
                        [],
                        positive_ids=[str(value) for value in record["positive_ids"]],
                    )
                )
        self.data = data
