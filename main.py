import os
import shutil
import pickle

from preprocess import preprocess
from torch import save


class TrainingHelper:
    def __init__(self):
        self.utils = TrainingHelperUtils()
        
        self.functions = {
            'Exit program': 'exit_program',
            'Preprocess data': 'preprocess',
            'Train a Network': 'training',
        }
        
        self.paths = {
            'raw data': 'Results',
            'datasets': 'Datasets',
            'model architectures': 'Models',
            'checkpoints': 'checkpoints',
        }
        
        self.main()
    
    def main(self):
        print('=== Welcome to the PhonoNet helper ===')
        print()
        print('Here are the available functions:')
        choice = self.select_from(self.functions)
        self.execute_function(self.functions[choice])
        print()
    
    def execute_function(self, function_name):
        function_choice = getattr(self, function_name)
        if callable(function_choice):
            function_choice()
        else:
            print("The function you are trying to call does not seem to exist.")
    
    def preprocess(self):
        
        # Selecting data to preprocess
        print('These are the folders in the (raw) Data folder. Which one do you want to use?')
        folders_list = self.utils.get_folders(self.paths['raw data'])
        data_name = self.utils.select_from(folders_list)
        
        # select dataset name
        print('These are the already exisiting datasets:')
        folders_list = self.utils.get_folders(self.paths['datasets'])
        for i in folders_list: print('- ', i)
        
        dataset_path = self.utils.get_new_name(data_name, self.paths['datasets'])
        
        print()
        print('Creating the dataset.')
        print()
        
        # get split        
        split = self.utils.get_split()
        
        # do the preprocessing
        train_dataset, val_dataset, test_dataset, scaler = preprocess(split, dataset_path)
        
        # create folder 
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        # save the scaler
        with open(os.path.join(dataset_path, 'scaler.pkl'), 'wb') as file:
            pickle.dump(scaler, file)
                
        # save datasets
        save(train_dataset, os.path.join(dataset_path, 'train.ds'))
        save(val_dataset, os.path.join(dataset_path, 'val.ds'))
        save(test_dataset, os.path.join(dataset_path, 'test.ds'))
        
        # # save info file
        # info_dict = {
            
        # }
    
    def training(self):
        print('training network wooosh (wip)')
    
    def exit_program(self):
        print('Exiting program')


class TrainingHelperUtils():
    def __init__(self):
        pass
    
    def select_from(self, choice_list):
        if type(choice_list) == dict:
            choice_list = list(choice_list.keys())
        
        # print the options
        for index, key in enumerate(choice_list):
            print(f'{index} - {key}')
        
        # Get the input number and make sure it is an allowed value
        while True:
            selected_nr = input('Choice: ')
            try: 
                selected_nr = int(selected_nr)
                if selected_nr in list(range(len(choice_list))): break
            except:
                pass
            print('Please type the number of the option you want to choose.')
        
        print()
        # return the selected key
        return choice_list[selected_nr]
    
    def get_folders(self, parent_folder):
        folders_list = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        return folders_list
    
    def get_new_name(self, name_recommendation, target_location):        
        folder_name = ''
        while not folder_name:
            print('What should the new folder be called (leave blank to use the same as the last step).')
            folder_name = input()
            if folder_name == '': folder_name = name_recommendation
            
            folder_path = os.path.join(target_location, folder_name)
            if os.path.isdir(target_location):                
                overwrite = input('The folder already exists. Do you want to overwrite it (yes/no): ')
                
                if overwrite == 'yes': shutil.rmtree(folder_path)
                else: folder_name = ''

        return folder_path

    def get_split(self):
        print('Enter the train, val, test split (example "0.7 0.1 0.2"):')
        while True:
            split = input()
            try:
                split = split.split(' ')
                if len(split) == 3:
                    split = [float(i) for i in split]
                    if sum(split) == 1:
                        break
            except:
                pass
            print('Please enter the split like this: 0.7 0.1 0.2 (must add up to one):')
            
        print()
        return split

if __name__ == "__main__":
    TrainingHelper()
    