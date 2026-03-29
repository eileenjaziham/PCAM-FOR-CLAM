import argparse
import gzip
import os
import shutil
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import get_encoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess PCam into CLAM bag features')
    parser.add_argument('--pcam_dir', type=str, default='pcam',
                        help='directory containing pcam/train, pcam/valid, pcam/test')
    parser.add_argument('--feat_dir', type=str, default='pcam_features',
                        help='output feature directory containing pt_files and optional h5_files')
    parser.add_argument('--csv_out', type=str, default='dataset_csv/pcam_clean.csv',
                        help='output csv for CLAM Generic_MIL_Dataset')
    parser.add_argument('--model_name', type=str, default='uni_v1',
                        choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'conch_v1_5'])
    parser.add_argument('--weights_path', type=str, default=None,
                        help='optional local checkpoint path for the encoder')
    parser.add_argument('--no_pretrained', action='store_true', default=False,
                        help='disable pretrained weights; useful in offline environments')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader workers per bag; keep 0 on CPU to avoid worker spawn overhead')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        choices=['train', 'valid', 'test'],
                        help='pcam splits to preprocess')
    parser.add_argument('--bag_label_mode', type=str, default='any',
                        choices=['any', 'majority', 'all'],
                        help='how to derive bag label from patch labels')
    parser.add_argument('--patches_per_bag', type=int, default=0,
                        help='number of patches per bag; <= 0 keeps one bag per WSI')
    parser.add_argument('--min_patches_per_bag', type=int, default=1,
                        help='drop sub-bags smaller than this threshold')
    parser.add_argument('--shuffle_within_wsi', action='store_true', default=False,
                        help='shuffle patches within each WSI before chunking into sub-bags')
    parser.add_argument('--label_source', type=str, default='y_h5',
                        choices=['y_h5', 'meta_tumor_patch', 'meta_center_tumor_patch'],
                        help='patch-level label source')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='directory for decompressed h5 cache; default uses split directory')
    parser.add_argument('--keep_decompressed', action='store_true', default=False,
                        help='keep decompressed h5 files when source is .gz')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing pt files and csv rows')
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_existing_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('None of these files exist: {}'.format(candidates))


def maybe_decompress_h5(path: str, cache_dir: str = None, keep_decompressed: bool = False) -> Tuple[str, bool]:
    if path.endswith('.h5'):
        return path, False

    if not path.endswith('.h5.gz'):
        raise ValueError('Unsupported file extension: {}'.format(path))

    base_name = os.path.basename(path[:-3])
    target_dir = cache_dir if cache_dir is not None else os.path.dirname(path)
    ensure_dir(target_dir)
    decompressed_path = os.path.join(target_dir, base_name)

    if not os.path.exists(decompressed_path):
        print('decompressing {} -> {}'.format(path, decompressed_path))
        with gzip.open(path, 'rb') as src, open(decompressed_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    should_cleanup = not keep_decompressed
    return decompressed_path, should_cleanup


def get_first_dataset(f: h5py.File, preferred_names: List[str]) -> h5py.Dataset:
    for name in preferred_names:
        if name in f:
            return f[name]

    dataset_names = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_names.append(name)

    f.visititems(visitor)
    if not dataset_names:
        raise KeyError('No datasets found in h5 file')
    return f[dataset_names[0]]


def load_meta(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    if 'Unnamed: 0' in meta.columns:
        meta = meta.drop(columns=['Unnamed: 0'])

    required_columns = {'wsi', 'coord_x', 'coord_y'}
    missing = required_columns - set(meta.columns)
    if missing:
        raise KeyError('Missing required columns in {}: {}'.format(meta_path, sorted(missing)))

    return meta


def get_patch_labels(meta: pd.DataFrame, y_h5_path: str, label_source: str) -> np.ndarray:
    if label_source == 'meta_tumor_patch':
        if 'tumor_patch' not in meta.columns:
            raise KeyError('meta csv does not contain tumor_patch column')
        return meta['tumor_patch'].astype(bool).to_numpy()

    if label_source == 'meta_center_tumor_patch':
        if 'center_tumor_patch' not in meta.columns:
            raise KeyError('meta csv does not contain center_tumor_patch column')
        return meta['center_tumor_patch'].astype(bool).to_numpy()

    with h5py.File(y_h5_path, 'r') as f:
        labels = get_first_dataset(f, ['y', 'labels'])[:]

    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(meta):
        raise ValueError('Label count {} does not match meta rows {}'.format(len(labels), len(meta)))
    return labels.astype(bool)


def bag_label_from_patch_labels(labels: np.ndarray, mode: str) -> str:
    labels = labels.astype(bool)
    if mode == 'any':
        is_positive = bool(labels.any())
    elif mode == 'majority':
        is_positive = bool(labels.mean() >= 0.5)
    elif mode == 'all':
        is_positive = bool(labels.all())
    else:
        raise NotImplementedError
    return 'positive' if is_positive else 'negative'


def split_indices_into_chunks(num_items: int, chunk_size: int) -> List[np.ndarray]:
    if chunk_size <= 0:
        return [np.arange(num_items, dtype=np.int64)]
    return [np.arange(start, min(start + chunk_size, num_items), dtype=np.int64)
            for start in range(0, num_items, chunk_size)]


class PCAMPatchBag(Dataset):
    def __init__(self, x_dataset: h5py.Dataset, patch_indices: np.ndarray, coords: np.ndarray, img_transforms):
        self.x_dataset = x_dataset
        self.patch_indices = patch_indices.astype(np.int64)
        self.coords = coords.astype(np.int32)
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        patch = self.x_dataset[self.patch_indices[idx]]
        img = Image.fromarray(patch)
        img = self.img_transforms(img)
        return {'img': img, 'coord': self.coords[idx]}


def compute_features_for_bag(x_dataset: h5py.Dataset, patch_indices: np.ndarray, coords: np.ndarray,
                             model, img_transforms, batch_size: int, num_workers: int) -> torch.Tensor:
    dataset = PCAMPatchBag(x_dataset=x_dataset,
                           patch_indices=patch_indices,
                           coords=coords,
                           img_transforms=img_transforms)
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': device.type == 'cuda'}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    bag_features = []

    for batch in loader:
        images = batch['img'].to(device, non_blocking=True)
        with torch.inference_mode():
            features = model(images).detach().cpu()
        bag_features.append(features)

    if not bag_features:
        raise ValueError('No features extracted for bag')

    return torch.cat(bag_features, dim=0)


def build_bag_index(meta: pd.DataFrame, patch_labels: np.ndarray, bag_label_mode: str, split_name: str,
                    patches_per_bag: int, min_patches_per_bag: int, shuffle_within_wsi: bool) -> List[Dict]:
    indexed_meta = meta.copy()
    indexed_meta['patch_label'] = patch_labels.astype(bool)
    indexed_meta['row_idx'] = np.arange(len(indexed_meta))

    bags = []
    grouped = indexed_meta.groupby('wsi', sort=False)
    for wsi_name, group in grouped:
        group = group.sort_values(['coord_y', 'coord_x']).reset_index(drop=True)
        if shuffle_within_wsi:
            group = group.sample(frac=1.0, random_state=0).reset_index(drop=True)

        chunks = split_indices_into_chunks(len(group), patches_per_bag)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_group = group.iloc[chunk].reset_index(drop=True)
            if len(chunk_group) < min_patches_per_bag:
                continue

            if patches_per_bag > 0:
                slide_id = '{}__{}__bag_{:04d}'.format(split_name, wsi_name, chunk_idx)
            else:
                slide_id = '{}__{}'.format(split_name, wsi_name)

            case_id = slide_id
            coords = chunk_group[['coord_x', 'coord_y']].to_numpy()
            patch_indices = chunk_group['row_idx'].to_numpy()
            bag_label = bag_label_from_patch_labels(chunk_group['patch_label'].to_numpy(), bag_label_mode)
            bags.append({
                'case_id': case_id,
                'slide_id': slide_id,
                'label': bag_label,
                'wsi': wsi_name,
                'patch_indices': patch_indices,
                'coords': coords,
            })
    return bags


def resolve_split_paths(split_dir: str, split_name: str) -> Tuple[str, str, str]:
    meta_path = os.path.join(split_dir, '{}_meta.csv'.format(split_name))
    x_path = find_existing_path([
        os.path.join(split_dir, '{}_x.h5'.format(split_name)),
        os.path.join(split_dir, '{}_x.h5.gz'.format(split_name)),
    ])
    y_path = find_existing_path([
        os.path.join(split_dir, '{}_y.h5'.format(split_name)),
        os.path.join(split_dir, '{}_y.h5.gz'.format(split_name)),
        os.path.join(split_dir, '{}_y.h5'.format(split_name), 'camelyonpatch_level_2_split_{}_y.h5'.format(split_name)),
    ])
    return meta_path, x_path, y_path


def preprocess_split(split_name: str, split_dir: str, feat_dir: str, model, img_transforms, args) -> List[Dict]:
    meta_path, x_path, y_path = resolve_split_paths(split_dir, split_name)
    meta = load_meta(meta_path)

    cache_dir = args.cache_dir
    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, split_name)

    x_h5_path, x_cleanup = maybe_decompress_h5(x_path, cache_dir=cache_dir, keep_decompressed=args.keep_decompressed)
    y_h5_path, y_cleanup = maybe_decompress_h5(y_path, cache_dir=cache_dir, keep_decompressed=args.keep_decompressed)

    try:
        patch_labels = get_patch_labels(meta, y_h5_path, args.label_source)
        bags = build_bag_index(
            meta,
            patch_labels,
            args.bag_label_mode,
            split_name,
            args.patches_per_bag,
            args.min_patches_per_bag,
            args.shuffle_within_wsi,
        )

        pt_dir = os.path.join(feat_dir, 'pt_files')
        ensure_dir(pt_dir)

        with h5py.File(x_h5_path, 'r') as x_file:
            x_dataset = get_first_dataset(x_file, ['x', 'imgs', 'images'])
            if len(x_dataset) != len(meta):
                raise ValueError('Image count {} does not match meta rows {}'.format(len(x_dataset), len(meta)))

            for bag in tqdm(bags, desc='{} bags'.format(split_name)):
                pt_path = os.path.join(pt_dir, '{}.pt'.format(bag['slide_id']))
                if os.path.exists(pt_path) and not args.overwrite:
                    continue

                features = compute_features_for_bag(
                    x_dataset=x_dataset,
                    patch_indices=bag['patch_indices'],
                    coords=bag['coords'],
                    model=model,
                    img_transforms=img_transforms,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                torch.save(features, pt_path)

        return [{'case_id': bag['case_id'], 'slide_id': bag['slide_id'], 'label': bag['label']} for bag in bags]
    finally:
        if x_cleanup and os.path.exists(x_h5_path):
            os.remove(x_h5_path)
        if y_cleanup and os.path.exists(y_h5_path):
            os.remove(y_h5_path)


def write_output_csv(rows: List[Dict], csv_out: str):
    ensure_dir(os.path.dirname(csv_out) or '.')
    df = pd.DataFrame(rows, columns=['case_id', 'slide_id', 'label'])
    df.to_csv(csv_out, index=False)
    print('wrote {} bag rows to {}'.format(len(df), csv_out))


def main():
    args = parse_args()
    ensure_dir(args.feat_dir)
    ensure_dir(os.path.join(args.feat_dir, 'pt_files'))

    print('inference device: {}'.format(device))
    if device.type != 'cuda' and args.num_workers > 0:
        print('CPU mode detected; overriding num_workers={} -> 0 for better throughput.'.format(args.num_workers))
        args.num_workers = 0

    model, img_transforms = get_encoder(
        args.model_name,
        target_img_size=args.target_patch_size,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.weights_path,
    )
    model = model.to(device)
    model.eval()

    all_rows = []
    for split_name in args.splits:
        split_dir = os.path.join(args.pcam_dir, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError('Missing split directory: {}'.format(split_dir))

        print('\nprocessing split: {}'.format(split_name))
        rows = preprocess_split(split_name, split_dir, args.feat_dir, model, img_transforms, args)
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError('No bag rows were generated')

    write_output_csv(all_rows, args.csv_out)


if __name__ == '__main__':
    main()
