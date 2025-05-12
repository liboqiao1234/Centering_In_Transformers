"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k
import os


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            
        try:
            train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        except:
            print("Failed to download Multi30k dataset, trying to load from local files...")
            train_data, valid_data, test_data = self.load_local_data()
            
        return train_data, valid_data, test_data

    def load_local_data(self):
        data_path = '.data/multi30k'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        # 创建示例数据文件
        train_en = os.path.join(data_path, 'train.en')
        train_de = os.path.join(data_path, 'train.de')
        valid_en = os.path.join(data_path, 'val.en')
        valid_de = os.path.join(data_path, 'val.de')
        test_en = os.path.join(data_path, 'test_2018_flickr.en')
        test_de = os.path.join(data_path, 'test_2018_flickr.de')
        
        # 创建TSV文件
        train_tsv = os.path.join(data_path, 'train.tsv')
        valid_tsv = os.path.join(data_path, 'val.tsv')
        test_tsv = os.path.join(data_path, 'test.tsv')
        
        # 读取并写入训练数据
        with open(train_en, 'r', encoding='utf-8') as f_en, \
             open(train_de, 'r', encoding='utf-8') as f_de, \
             open(train_tsv, 'w', encoding='utf-8') as f_tsv:
            for en_line, de_line in zip(f_en, f_de):
                f_tsv.write(f"{en_line.strip()}\t{de_line.strip()}\n")
                
        # 读取并写入验证数据
        with open(valid_en, 'r', encoding='utf-8') as f_en, \
             open(valid_de, 'r', encoding='utf-8') as f_de, \
             open(valid_tsv, 'w', encoding='utf-8') as f_tsv:
            for en_line, de_line in zip(f_en, f_de):
                f_tsv.write(f"{en_line.strip()}\t{de_line.strip()}\n")
                
        # 读取并写入测试数据
        with open(test_en, 'r', encoding='utf-8') as f_en, \
             open(test_de, 'r', encoding='utf-8') as f_de, \
             open(test_tsv, 'w', encoding='utf-8') as f_tsv:
            for en_line, de_line in zip(f_en, f_de):
                f_tsv.write(f"{en_line.strip()}\t{de_line.strip()}\n")
            
        # 使用TabularDataset加载数据
        train_data = TabularDataset(
            path=train_tsv,
            format='tsv',
            fields=[('src', self.source), ('trg', self.target)]
        )
        
        valid_data = TabularDataset(
            path=valid_tsv,
            format='tsv',
            fields=[('src', self.source), ('trg', self.target)]
        )
        
        test_data = TabularDataset(
            path=test_tsv,
            format='tsv',
            fields=[('src', self.source), ('trg', self.target)]
        )
        
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True
        )
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
