import os,sys
import os.path
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import urllib.request
from PIL import Image
import pickle
import utils

########################################################################################################################


def _load_dataset(loader, transforms, name, ncla, size=None, expand_channels=False):
    mean=[x/255 for x in mean]
    std=[x/255 for x in std]
    dat={}
    
    # check if resize transform should be applied
    if size is not None:
        pass

    # load the datasets
    dat['train']=loader('../dat/',train=True,download=True,transform=transforms())
    dat['test']=loader('../dat/',train=False,download=True,transform=transforms())
    data={}
    data['name']=name
    data['ncla']=ncla
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        data[s]={'x': [],'y': []}
        for image,target in loader:
            # check if channels should be expanded
            if expand_channels is True:
                image=image.expand(1,3,image.size(2),image.size(3)) # Create 3 equal channels
            # add to dataset
            data[s]['x'].append(image)
            data[s]['y'].append(target.numpy()[0])
    return data

def _load_cifar10():
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])

    return _load_dataset(datasets.CIFAR10, tfs, "cifar10", 10)

def _load_cifar100():
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])

    return _load_dataset(datasets.CIFAR100, tfs, "cifar100", 100)

def _load_mnist():
    #mean=(0.1307,) # Mean and std without including the padding
    #std=(0.3081,)
    mean=(0.1,) # Mean and std including the padding
    std=(0.2752,)
    tfs = transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)])

    return _load_dataset(datasets.MNIST, tfs, "mnist", 10, expand_channels=True)

def _load_fashion_mnist():
    mean=(0.2190,) # Mean and std including the padding
    std=(0.3318,)
    tfs = transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)])

    return _load_dataset(datasets.FashionMNIST, tfs, "fashion_mnist", 10, expand_channels=True)

def _load_not_mnist():
    mean=(0.4254,)
    std=(0.4501,)
    tfs = transforms.Compose([transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)])

    return _load_dataset(notMNIST, tfs, "not_mnist", 10, expand_channels=True)

def _load_svhn():
    mean=[0.4377,0.4438,0.4728]
    std=[0.198,0.201,0.197]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    
    return _load_dataset(datasets.SVHN, tfs, "svhn", 10)

def _load_traffic_signs():
    mean=[0.3398,0.3117,0.3210]
    std=[0.2755,0.2647,0.2712]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    
    return _load_dataset(TrafficSigns, tfs, "traffic_signs", 43)

def _load_cub200():
    mean=[x/255 for x in [90.6379,93.2626,80.7344]]
    std=[x/255 for x in [71.4966,71.0943,72.5651]]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    
    return _load_dataset(Cub200, tfs, "cub200", 200)

def _load_facescrub():
    mean=[0.5163,0.5569,0.4695]
    std=[0.2307,0.2272,0.2479]
    tfs = lambda: transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    
    return _load_dataset(Facescrub, tfs, "facescrub", 100)


def get(seed=0,fixed_order=False,pc_valid=0.15,num_data=8,sample=False):
    data={}
    taskcla=[]
    # output size for training (channels first)
    # TODO: update resize options
    size=[3,32,32]

    # create dataset list
    idata=np.arange(num_data)
    if sample is True:
        idata = np.random.choice(np.arange(9), num_data, replace=False)
    # TODO: update dataset collection
    # shuffle if activated
    if not fixed_order:
        idata=list(shuffle(idata,random_state=seed))
    print('Task order =',idata)

    # iterate through all data and save in binary format (for faster loading)
    if not os.path.isdir('../dat/binary_mixture/'):
        os.makedirs('../dat/binary_mixture')
        # Pre-load
        for n,idx in enumerate(idata):
            if idx==0:
                # CIFAR10
                data[n] = _load_cifar10()

            elif idx==1:
                # CIFAR100
                data[n] = _load_cifar100()

            elif idx==2:
                # MNIST
                data[n]= _load_mnist()

            elif idx == 3:
                # SVHN
                data[n] = _load_svhn()

            elif idx == 4:
                # FashionMNIST
                data[n] = _load_fashion_mnist()

            elif idx == 5:
                # TrafficSigns
                data[n] = _load_traffic_signs()

            elif idx == 6:
                # Facescrub 100 faces
                data[n] = _load_facescrub()
                
            elif idx == 7:
                # notMNIST A-J letters
                data[n] = _load_not_mnist()

            elif idx == 8:
                # CUB 200
                data[n] = _load_cub200()

            else:
                print('ERROR: Undefined data set',n)
                sys.exit()
            #print(n,data[n]['name'],data[n]['ncla'],len(data[n]['train']['x']))

            # TODO: apply complexity before saving (for easier filter later on)

            # "Unify" and save
            for s in ['train','test']:
                data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
                data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'y.bin'))

    else:

        # Load binary files
        for n,idx in enumerate(idata):
            data[n] = dict.fromkeys(['name','ncla','train','test'])
            if idx==0:
                data[n]['name']='cifar10'
                data[n]['ncla']=10
            elif idx==1:
                data[n]['name']='cifar100'
                data[n]['ncla']=100
            elif idx==2:
                data[n]['name']='mnist'
                data[n]['ncla']=10
            elif idx==3:
                data[n]['name']='svhn'
                data[n]['ncla']=10
            elif idx==4:
                data[n]['name']='fashion_mnist'
                data[n]['ncla']=10
            elif idx==5:
                data[n]['name']='traffic_signs'
                data[n]['ncla']=43
            elif idx==6:
                data[n]['name']='facescrub'
                data[n]['ncla']=100
            elif idx==7:
                data[n]['name']='not_mnist'
                data[n]['ncla']=10
            elif idx==8:
                data[n]['name']='cub200'
                data[n]['ncla']=200
            else:
                print('ERROR: Undefined data set',n)
                sys.exit()

            # Load
            for s in ['train','test']:
                data[n][s]={'x':[],'y':[]}
                data[n][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'x.bin'))
                data[n][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_mixture'),'data'+str(idx)+s+'y.bin'))

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################

class FashionMNIST(datasets.MNIST):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

########################################################################################################################

class TrafficSigns(torch.utils.data.Dataset):
    """`German Traffic Signs <http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'lab 2 data/train.p'
        testing_file = 'lab 2 data/test.p'
        if train:
            with open(os.path.join(root,training_file), mode='rb') as f:
                train = pickle.load(f)
            self.data = train['features']
            self.labels = train['labels']
        else:
            with open(os.path.join(root,testing_file), mode='rb') as f:
                test = pickle.load(f)
            self.data = test['features']
            self.labels = test['labels']

        self.data = np.transpose(self.data, (0, 3, 1, 2))
        #print(self.data.shape); sys.exit()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class Facescrub(torch.utils.data.Dataset):
    """Subset of the Facescrub cropped from the official Megaface challenge page: http://megaface.cs.washington.edu/participate/challenge.html, resized to 38x38

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "facescrub_100.zip"
        self.url = "https://github.com/nkundiushuti/facescrub_subset/blob/master/data/facescrub_100.zip?raw=true"

        fpath=os.path.join(root,self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'facescrub_train_100.pkl'
        testing_file = 'facescrub_test_100.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
            """
            print(self.data.shape)
            print(self.data.mean())
            print(self.data.std())
            print(self.labels.max())
            #"""
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class notMNIST(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)


    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


########################################################################################################################

class Cub200(torch.utils.data.Dataset):
    '''Loads the CUB200-2011 Dataset.

    Reference: Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001.
    '''
    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "CUB_200_2011.tgz"
        self.url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()
        
        # load the data information
        root = os.path.join(root, 'CUB_200_2011')
        train_test = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=' ', header=None, names=['id', 'is_train'])
        images = pd.read_csv(os.path.join(root, "images.txt"), sep=' ', header=None, names=['id', 'path'])
        cls_label = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=' ', header=None, names=['id', 'class_id'])
        cls_names = pd.read_csv(os.path.join(root, "classes.txt"), sep=' ', header=None, names=['class_id', 'class_name'])

        # merge the data
        min_cid = np.min(cls_names['class_id'])
        cls_label['class_id'] = cls_label['class_id'] - min_cid
        cls_names['class_id'] = cls_names['class_id'] - min_cid
        df_images = pd.merge(train_test, images, on='id')
        df_images = pd.merge(df_images, cls_label, on='id')
        num_classes = len(cls_names.index)
        cid = np.array(cls_names['class_id'])

        # shuffle the dataset
        # TODO: updated random state?
        df_images = df_images.sample(frac=1, random_state=123)

        # select the correct data
        if train is True:
            df_images = df_images[df_images['is_train'] == 1]
        else:
            df_images = df_images[df_images['is_train'] == 0]
        
        # load into data
        tmp_data = []
        tmp_lbls = []
        for idx, row in df_images.iterrows():
            # load the image and label
            img = Image.open(os.path.join(root, 'images', row['path'])).convert("RGB")
            cla = row['class_id']
            
            # resize the image to common input
            img, scale = utils.resize_and_pad(img, (32, 32), 'fit_center')
            
            # append data
            tmp_data.append(np.array(img).astype('float'))
            tmp_lbls.append(cla)
        
        # combine data
        self.data = np.stack(tmp_data, axis=0)
        self.labels = np.stack(tmp_lbls, axis=0)

        # convert to channels first
        self.data = np.transpose(self.data, (0, 3, 1, 2))

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        import tarfile
        tar_ref = tarfile.open(fpath, 'r:gz')
        tar_ref.extractall(root)
        tar_ref.close()
        # CHECK: check if zip needs to be extracted

########################################################################################################################
