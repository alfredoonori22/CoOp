import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing
from .oxford_pets import OxfordPets

dict_age_to_label = {
    '0-2': 'infant',
    '3-9': 'child',
    '10-19': 'teen',
    '20-29': 'twenties',
    '30-39': 'thirties',
    '40-49': 'forties',
    '50-59': 'fifties',
    '60-69': 'sixties',
    'more than 70': 'senior'
}

dict_age_to_number = {'infant': 0,
                      'child': 1,
                      'teen': 2,
                      'twenties': 3,
                      'thirties': 4,
                      'forties': 5,
                      'fifties': 6,
                      'sixties': 7,
                      'senior': 8}

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
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        
        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.image_dir, cfg.FAIRFACECLASS)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            # TODO: ho modificato il path 
            preprocessed = os.path.join(self.split_fewshot_dir, f"age_shot_{num_shots}-seed_{seed}.pkl")
            
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
                    item = Datum(impath=impath, label=int(dict_age_to_number[dict_age_to_label[classname[0]]]), classname=dict_age_to_label[classname[0]])
                elif class_label == 'gender':
                    item = Datum(impath=impath, label=int(dict_gender_to_number[classname[1]]), classname=classname[1])
                elif class_label == 'race':
                    item = Datum(impath=impath, label=int(dict_race_to_number[classname[2]]), classname=classname[2])
                else: 
                    print(f'FairFace class {class_label} does not exist')
                    raise Exception('FairFace Class Error')
                
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = train[int(len(train)-0.1*len(train)):]
        train = train[:int(len(train)-0.1*len(train))]
        test = _convert(split["valid"])

        return train, val, test
