import numpy as np
import torch
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
from img_text_composition_models import TIRG
class ClothesRetrieval(torch.utils.data.Dataset):

    def __init__(self, model:TIRG, dataset_path):
        super().__init__()

        print('Initiating...')
        self.image_paths = []
        self.image_captions = []
        self.path2id = {}
        self.model = model        

        # ----------------------------------------------------------------------------------------------------------------------------
        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                                         '&', 'andmark').replace('*', 'starmark')

        label_folder = os.path.join(dataset_path, 'labels')
        label_files = glob(label_folder + '*.txt')
        label_files = [label_file for label_file in label_files if 'test' in label_file]

        index = 0
        images = []
        for label_file in tqdm(label_files):
            with open(label_file) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')                

                self.image_paths += line[0]
                self.image_captions += [caption_post_process(line[2])]
                images += [self._load_image(line[0])]
                self.path2id[line[0]] = index
                index += 1

        images = [torch.from_numpy(image).float() for image in images]
        images = torch.stack(images).float()
        images = torch.autograd.Variable(images).cuda()
        image_features = np.array(self.model.extract_img_feature(images).data.cpu().numpy())
        for i in range(image_features.shape[0]):
            image_features[i, :] /= np.linalg.norm(image_features[i, :])
        self.image_features = image_features

        print('===Finish===')
        print(f' - Number of image infos: {len(self.image_infos)} \n - Number of image features: {len(self.image_features)}')

    def _load_image(self, path):
            image = Image.open(path)
            image = image.convert('RGB')
            return image

    def retrieve_image(self, image_path, modify, number_of_outputs=3):
        image = self._load_image(image_path)
        images = [torch.from_numpy(image).float()]
        images = torch.stack(images).float()
        images = torch.autograd.Variable(images).cuda()

        modifies = [modify]
        queries = self.model.compose_img_text(images, modifies).data.cpu().numpy() # shape = (1, None)
        queries /= np.linalg.norm(queries)
        sims = np.squeeze(queries.dot(self.image_features.T))
        index_results = np.argsort(-sims)[:110]
        sorted_image_paths = [self.image_paths[index] for index in index_results][:number_of_outputs]

        return sorted_image_paths

    def retrieve_multi_images(self, image_path_list, modify_list, number_of_outputs=3):
        images = [self._load_image(image_path) for image_path in image_path_list]
        images = [torch.from_numpy(image).float() for image in images]
        images = torch.stack(images).float()
        images = torch.autograd.Variable(images).cuda()

        queries = self.model.compose_img_text(images, modify_list).data.cpu().numpy()
        for i in range(queries.shape[0]):
            queries[i, :] /= np.linalg.norm(queries[i, :])

        nn_result = []
        for i in tqdm(range(queries.shape[0])):
            sims = queries[i:(i+1), :].dot(self.image_features.T)
            nn_result.append(np.argsort(-sims[0, :])[:110])

        nn_result = [[self.image_paths[nn] for nn in nns] for nns in nn_result]

        return nn_result