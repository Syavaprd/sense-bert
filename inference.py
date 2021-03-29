import os
import pandas as pd
from pathlib import Path
from sensebert import SenseBert
import tensorflow as tf
from tqdm import tqdm
import sys
import numpy as np

def idx_word2idx_ch(sent: str, idx_word: int):
  sent = sent.split(' ')
  start = sum(len(w) for w in sent[:idx_word]) + idx_word  # + idx_word - прибавляем количество пробелов между словами
  end = start + len(sent[idx_word])
  return start, end

def load_data(names_datasets=None, only_with_gold: bool =False, include_trial: bool = True, dataset_path=None):
  """
  :param names_datasets: dataset name or list of datasets names, e.g. ['mcl-wic'], by default = list of all datasets
  :param only_with_gold: if True, load data with gold labels only; False - load all data
  :param include_trial: if True, include trial datasets in returning data
  :return: dict{split_name_dataset: pd.DataFrame}, e.g. {'train_wic': df1, 'dev_mcl-wic_en-ru': df2}
           format split_name_dataset: <split>_<name_dataset>[_<params_dataset>], at [] - optional
  """

  if names_datasets is None:
    names_datasets = ['wic', 'mcl-wic']
  elif type(names_datasets) == str:
    names_datasets = [names_datasets]
  if dataset_path is None:
      datasets_dir = './datasets'
  else:
      datasets_dir = dataset_path

  full_splits = {'dev', 'test', 'train', 'training'}  # 'training' in mcl-wic; 'train' in wic
  if include_trial:
    full_splits.add('trial')
  full_splits_mcl_wic = {'crosslingual', 'multilingual'}
  data = {}
  for name_data in names_datasets:
    splits = set(os.listdir(f'{datasets_dir}/{name_data}')) & full_splits
    for split in splits:
      if name_data == 'wic':
        path_gold = f'{datasets_dir}/{name_data}/{split}/{split}.gold.txt'
        path_data = f'{datasets_dir}/{name_data}/{split}/{split}.data.txt'
        is_gold_exits = os.path.exists(path_gold)
        if only_with_gold and (not is_gold_exits):
          continue
        df = pd.read_csv(path_data, sep='\t', names=['lemma', 'pos', 'idxs', 'sentence1', 'sentence2'])
        df.idxs = df.idxs.apply(lambda s: [int(idx) for idx in s.split('-')])
        assert set(df.pos).issubset({'N', 'V'}), f"Unexpected part of speech in df.pos = {set(df.pos)}, available - 'N' and 'V'"
        df.pos = df.pos.apply(lambda p: 'NOUN' if p == 'N' else 'VERB')
        for num_sent in (1, 2):
          df[f'start{num_sent}'] = df.apply(lambda r: idx_word2idx_ch(r[f'sentence{num_sent}'], r.idxs[num_sent - 1]),
                                            axis=1)
          df[f'end{num_sent}'] = df[f'start{num_sent}'].apply(lambda pair: pair[1])
          df[f'start{num_sent}'] = df[f'start{num_sent}'].apply(lambda pair: pair[0])
        df = df.drop(columns=['idxs'])
        df['id'] = [f'{split}.{idx}' for idx in range(df.shape[0])]

        if is_gold_exits:
          with open(path_gold, 'r') as file:
            gold_labels = file.read().split()
          df['tag'] = gold_labels

        data[f'{split}_{name_data}'] = df
      elif name_data == 'mcl-wic':
        path_mcl_wic = f'{datasets_dir}/{name_data}/{split}'
        splits_mcl_wic = set(os.listdir(path_mcl_wic)) & full_splits_mcl_wic  # set of available splits, all possible splits at <full_splits_mcl_wic>
        for split_mcl_wic in splits_mcl_wic:
          name_files = [name.split('.')[1] for name in os.listdir(f'{path_mcl_wic}/{split_mcl_wic}') if name.startswith(f'{split}.') and
                (name.endswith('.data') or name.endswith('.gold'))]  # оставляем 'lang1-lang2' из файлов вида <split>.<что-то>.[gold/data]
          name_files = set(name_files)
          for langs in name_files:
            path_gold = f'{path_mcl_wic}/{split_mcl_wic}/{split}.{langs}.gold'
            is_gold_exits = os.path.exists(path_gold)
            if only_with_gold and (not is_gold_exits):
              continue

            df = pd.read_json(f'{path_mcl_wic}/{split_mcl_wic}/{split}.{langs}.data', orient='records')
            if is_gold_exits:
              gold_df = pd.read_json(path_gold, orient='records')
              df = df.merge(gold_df)
            data[f'{split}_{name_data}_{langs}'] = df
      else:
        raise ValueError("Wrong name dataset! Use 'mcl-wic' or 'wic'.")
  return data


def preprocess_dataframe(df):
    df['Word1'] = [row.sentence1[row.start1:row.end1] for index, row in df.iterrows()]
    df['Word2'] = [row.sentence2[row.start2:row.end2] for index, row in df.iterrows()]
    df['pos1'] = [len(row.sentence1[:row.start1].split(' ')) for index, row in df.iterrows()]
    df['pos2'] = [len(row.sentence2[:row.start2].split(' ')) for index, row in df.iterrows()]
    #for index, row in df.iterrows():
    #    print(row.lemma)
    #    print(row.sentence1.split(' ')[row.pos1 - 1])

def get_accuracy(df, model):
    res = []
    def inference(df):
        with  tf.compat.v1.Session() as session:
            sensebert_model = SenseBert(f"{model}", session=session)  # or sensebert-large-uncased
            for index, row in tqdm(df.iterrows()):
                input_ids, input_mask = sensebert_model.tokenize([row.sentence1, row.sentence2])
                model_outputs = sensebert_model.run(input_ids, input_mask)
                contextualized_embeddings, mlm_logits, supersense_logits = model_outputs
                super1 = np.argmax(supersense_logits[0][row.pos1])
                super2 = np.argmax(supersense_logits[1][row.pos2 ])
                if super1 == super2:
                    res.append(1)
                else:
                    res.append(0)
    preprocess_dataframe(df)
    inference(df)
    if 'tag' in df.columns:
        k = 0
        for i in range(df.shape[0]):
            tag = df.tag[i]
            if tag == 'T' and res[i] == 1:
                k += 1
            elif tag == 'F' and res[i] == 0:
                k += 1
            else:
                continue
        accuracy = k / df.shape[0]
        print(f"accuracy===={accuracy}")
    else:
        return res
    return

def save_predictions(df, path_save: str):
  """
  :param df: pd.DataFrame with columns 'pred' and 'id' (unique id of instance)


  :param path_save: file path to save predictions in JSON-file
  """
  assert 'id' in df and 'pred' in df, 'Expected columns "id" and "pred" in the DataFrame df'
  df[['id', 'pred']].to_json(path_save, orient='records')

if __name__ == "__main__":
    competition = sys.argv[1]
    dataset = sys.argv[2]
    model = sys.argv[3]
    if len(sys.argv) > 4:
        dataset_path = sys.argv[4]
    d = load_data(competition, dataset_path=dataset_path)
    df = d[dataset]
    get_accuracy(df, model) 
