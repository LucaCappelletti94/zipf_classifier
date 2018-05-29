"""The builder builds a zipf for every options set in the options.json file."""
import gc
from json import dump, load
from os import makedirs
from os.path import dirname, exists, join

from zipf.factories import ZipfFromDir


def _load_options_set():
    path = join(dirname(__file__), 'options.json')
    with open(path, "r") as f:
        return load(f)


def _build_zipfs(full_path, names):
    save_path = "/Users/lucacappelletti/Datasets/zipf_datasets/zipfs"
    if not exists(save_path):
        makedirs(save_path)
    options_set = _load_options_set()
    for i, options in enumerate(options_set):
        factory = ZipfFromDir(options=options, use_cli=True)
        dir_path = save_path + "/%s" % (i)
        options_path = dir_path + "/options.json"
        if not exists(dir_path):
            makedirs(dir_path)
        with open(options_path, "w") as f:
            dump(options, f)
        for name in names:
            data_path = full_path % name
            file_path = dir_path + "/%s.json" % (name)
            if not exists(file_path):
                factory.run(paths=data_path, extensions=[
                    "txt"]).save(file_path)
            gc.collect()


if __name__ == "__main__":
    _build_zipfs(
        "/Users/lucacappelletti/Datasets/zipf_datasets/for_zipf/%s",
        [
            "wikipedia",
            # "ricette_zafferano",
            # "personal_trainer"
        ]
    )
