from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os
import logging
import glob
import random

logger = logging.getLogger(__name__)


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataset_folder = opt.dataroot
        self.folder_name = opt.img_folder
        self.extension = 'jpg'
        # self.mask_extension = 'png'
        # self.mask_folder_name = 'sil_mad'
        self.random_view = True
        self.is_generation = opt.is_generation

        # import pdb; pdb.set_trace()

        if opt.classes == '03001627':
            self.classes = ['03001627']
        elif opt.classes == 'mix':
            self.classes = ['03001627', '02691156', '03636649']
        else:
            raise ValueError('classes must be either 03001627 or mix')

        # Get all models
        self.models = []
        for c in self.classes:
            subpath = os.path.join(self.dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, opt.phase + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            self.models += [
                {'category': c, 'model': m}
                for m in models_c
            ]

        # self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        import pudb; pu.db

        category = self.models[idx]['category']
        model = self.models[idx]['model']

        model_path = os.path.join(self.dataset_folder, category, model)

        folder = os.path.join(model_path, self.folder_name)
        folder_base = os.path.join(folder, 'base')
        folder_bias = os.path.join(folder, 'bias')
        files = sorted(glob.glob(os.path.join(folder_base, '*.%s' % self.extension)))
        files_bias = sorted(glob.glob(os.path.join(folder_bias, '*.%s' % self.extension)))
        files.extend(files_bias)

        # mask_folder = os.path.join(model_path, self.mask_folder_name)
        # mask_base = os.path.join(mask_folder, 'base')
        # mask_bias = os.path.join(mask_folder, 'bias')
        # masks = sorted(glob.glob(os.path.join(mask_base, '*.%s' % self.mask_extension)))
        # masks_bias = sorted(glob.glob(os.path.join(mask_bias, '*.%s' % self.mask_extension)))
        # masks.extend(masks_bias)
        if not self.is_generation:
            if self.random_view:
                idx_img = random.randint(0, len(files)-1)
            else:
                idx_img = 0
            filename = files[idx_img]
            # maskname = masks[idx_img]

            image = Image.open(filename).convert('RGB')
            # mask = Image.open(maskname).convert('RGB')

            # A_path = self.A_paths[index]
            # A_img = Image.open(A_path).convert('RGB')
            image = self.transform(image)
            return {'A': image, 'A_paths': filename}
        else:
            # images = []
            # filenames = []
            # import pdb; pdb.set_trace()
            img_dicts = []
            for i in range(len(files)):
                filename = files[i]
                image = Image.open(filename).convert('RGB')
                image = self.transform(image)
                img_dict = {'A': image, 'A_paths': filename}
                img_dicts.append(img_dict)
            return img_dicts
                # images.append(image)
                # filenames.append(filename)
                # return {'A': images, 'A_paths': filenames}



    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.models)
