import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps 
import glob
import random
import logging
from util.util import sparse_label


logger = logging.getLogger(__name__)


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
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
        self.mask_extension = 'png'
        self.mask_folder_name = 'sil_mad'
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

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # import pudb; pu.db
        
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        # import pudb; pu.db

        model_path = os.path.join(self.dataset_folder, category, model)

        if not self.folder_name == 'human':
            folder = os.path.join(model_path, self.folder_name)
            folder_base = os.path.join(folder, 'base')
            folder_bias = os.path.join(folder, 'bias')
            files = sorted(glob.glob(os.path.join(folder_base, '*.%s' % self.extension)))
            files_bias = sorted(glob.glob(os.path.join(folder_bias, '*.%s' % self.extension)))
            files.extend(files_bias)

            mask_folder = os.path.join(model_path, self.mask_folder_name)
            mask_base = os.path.join(mask_folder, 'base')
            mask_bias = os.path.join(mask_folder, 'bias')
            masks = sorted(glob.glob(os.path.join(mask_base, '*.%s' % self.mask_extension)))
            masks_bias = sorted(glob.glob(os.path.join(mask_bias, '*.%s' % self.mask_extension)))
            masks.extend(masks_bias)
        
        else:
             folder = os.path.join(model_path, self.folder_name)
             files = sorted(glob.glob(os.path.join(folder, '*.%s' % self.extension)))
             mask_folder = os.path.join(model_path, self.mask_folder_name)
             mask_base = os.path.join(mask_folder, 'base')
             masks = sorted(glob.glob(os.path.join(mask_base, '*.%s' % self.mask_extension)))

        if not self.is_generation:
            if self.random_view:
                idx_img = random.randint(0, len(files)-1)
            else:
                idx_img = 0
            filename = files[idx_img]
            maskname = masks[idx_img]

            image = Image.open(filename).convert('RGB')
            mask = Image.open(maskname).convert('RGB')

            mask = ImageOps.invert(mask)
            # apply the same transform to both A and B
            transform_params = get_params(self.opt, image.size)
            image_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

            image = image_transform(image)
            mask = mask_transform(mask)
            label = sparse_label(mask)

            return {'A': image, 'B': mask, 'A_paths': filename, 'B_paths': maskname, 'label': label}
        else:
            img_dicts = []
            for i in range(len(files)):
                filename = files[i]
                maskname = masks[i]
                image = Image.open(filename).convert('RGB')
                mask = Image.open(maskname).convert('RGB')
                mask = ImageOps.invert(mask)
                transform_params = get_params(self.opt, image.size)
                image_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
                mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
                image = image_transform(image)
                mask = mask_transform(mask)
                label = sparse_label(mask)
                img_dict = {'A': image, 'B': mask, 'A_paths': filename, 'B_paths': maskname, 'label': label}
                img_dicts.append(img_dict)
            return img_dicts


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.models)
