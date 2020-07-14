from PyInstaller.utils.hooks import collect_data_files
hiddenimports = ['sklearn.utils.class_weight',
                 'sklearn.utils.validation',
                 'sklearn.utils.fixes',
                 'scipy.stats']

datas = collect_data_files('sklearn')