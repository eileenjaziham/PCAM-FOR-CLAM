import os


TASK_CONFIGS = {
    'task_1_tumor_vs_normal': {
        'n_classes': 2,
        'csv_path': 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
        'features_subdir': 'tumor_vs_normal_resnet_features',
        'label_dict': {'normal_tissue': 0, 'tumor_tissue': 1},
        'patient_strat_train': False,
        'patient_strat_split': True,
    },
    'task_2_tumor_subtyping': {
        'n_classes': 3,
        'csv_path': 'dataset_csv/tumor_subtyping_dummy_clean.csv',
        'features_subdir': 'tumor_subtyping_resnet_features',
        'label_dict': {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
        'patient_strat_train': False,
        'patient_strat_split': True,
        'patient_voting': 'maj',
        'requires_subtyping': True,
    },
    'task_pcam': {
        'n_classes': 2,
        'csv_path': 'dataset_csv/pcam_clean.csv',
        'features_subdir': 'pcam_features',
        'label_dict': {'negative': 0, 'positive': 1},
        'patient_strat_train': False,
        'patient_strat_split': False,
    },
}


TASK_CHOICES = list(TASK_CONFIGS.keys())


def get_task_config(task):
    if task not in TASK_CONFIGS:
        raise NotImplementedError
    return TASK_CONFIGS[task]


def build_dataset_kwargs(task, data_root_dir=None, for_splits=False):
    config = get_task_config(task)
    dataset_kwargs = {
        'csv_path': config['csv_path'],
        'shuffle': False,
        'print_info': True,
        'label_dict': config['label_dict'],
        'patient_strat': config['patient_strat_split'] if for_splits else config['patient_strat_train'],
        'ignore': [],
    }

    if 'patient_voting' in config:
        dataset_kwargs['patient_voting'] = config['patient_voting']

    if data_root_dir is not None:
        dataset_kwargs['data_dir'] = os.path.join(data_root_dir, config['features_subdir'])

    return dataset_kwargs
