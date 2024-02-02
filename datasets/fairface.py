import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing
from .oxford_pets import OxfordPets

dict_age_to_number = {'0-2': 0,
                     '3-9': 1,
                     '10-19': 2,
                     '20-29': 3,
                     '30-39': 4,
                     '40-49': 5,
                     '50-59': 6,
                     '60-69': 7,
                     'more than 70': 8}

dict_gender_to_number = {'Male': 0,
                         'Female': 1}

dict_race_to_number = {'White' : 0,
                       'Black': 1,
                       'Latino_Hispanic': 2,
                       'East Asian' : 3,
                       'Southeast Asian' : 4,
                       'Indian' : 5,
                       'Middle Eastern' : 6}
@DATASET_REGISTRY.register()
class FairFace(DatasetBase):
    
    dataset_dir = "FairFace"

    def __init__(self, cfg):
        root = os.path.abspath('/work/tesi_aonori/CoOp_datasets/')
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, "labels.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"{cfg.FAIRFACECLASS}_split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        
        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.image_dir, cfg.FAIRFACECLASS)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"{cfg.FAIRFACECLASS}_shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
        super().__init__(train_x=train, val=val, test=test)

        

def read_split(filepath, path_prefix, class_label):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                if class_label == 'age':
                    item = Datum(impath=impath, label=int(dict_age_to_number[classname[0]]), classname=classname[0])
                elif class_label == 'gender':
                    item = Datum(impath=impath, label=int(dict_gender_to_number[classname[1]]), classname=classname[1])
                elif class_label == 'race':
                    item = Datum(impath=impath, label=int(dict_race_to_number[classname[2]]), classname=classname[2])
                else:
                    raise Exception(f'FairFace class {class_label} does not exist')

                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)

        if class_label == "gender":
            train = _convert(split["train"])
            test = _convert(split["valid"])
        elif class_label == "race":
            train = _convert(split["train"])
            test = _convert(split["valid"])
        elif class_label == "age":
            train = _convert(split["train"])
            test = _convert(split["valid"])
        else:
            raise Exception(f'FairFace class {class_label} does not exist')

        val = train[int(0.9*len(train)):]
        train = train[:int(0.9*len(train))]

        return train, val, test
