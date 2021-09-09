import torch
from torch import nn
import torch.nn.functional as F
import torchmeta



class MyLinear(nn.Module):
    '''
    Implements linear layer that handles inputs with different shapes
    input_shape: (*shape, inp_dim)
    output_shape: (*shape, out_dim)
    '''
    def __init__(self, inp_dim, out_dim, bias=True):
        super(MyLinear, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(inp_dim, out_dim, bias=bias)

    def forward(self, x):
        shape = x.shape
        assert shape[-1] == self.inp_dim
        x = x.view(-1, shape[-1])

        x = self.linear(x)

        return x.view(*shape[:-1], x.shape[-1])



class MetaLearningModel(nn.Module):
    def __init__(self, e2e):
        super(MetaLearningModel, self).__init__()
        self.e2e = e2e
        self.classifier = self.e2e

    def forward(self, x):
        reps = self.representation(x)
        if not self.e2e:
            return reps
        return self.classifier(self.representation(x))

    def representation(self, x):
        pass



class SineModel(MetaLearningModel):
    def __init__(self, hid_dim, e2e=True, classifier=False):
        super(SineModel, self).__init__(e2e)
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(1, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        if self.classifier or classifier:
            self.classifier = MyLinear(self.hid_dim, 1)
        else:
            self.classifier = None
        
    def representation(self, x):
        shape = x.shape
        assert shape[-1:] == (1,)
        x = x.view(-1, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x.view(*shape[:-1], x.shape[-1])



class LinearRepModel(MetaLearningModel):
    def __init__(self, inp_dim, hid_dim, e2e=True, classifier=False):
        super(LinearRepModel, self).__init__(e2e)
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(inp_dim, hid_dim, bias=False)

        if self.classifier or classifier:
            self.classifier = MyLinear(self.hid_dim, 1)
        else:
            self.classifier = None
        
    def representation(self, x):
        shape = x.shape
        assert shape[-1:] == (self.inp_dim,)
        x.view(-1, self.inp_dim)

        x = self.fc1(x)
        return x.view(*shape[:-1], x.shape[-1])



class OmniglotModel(MetaLearningModel):
    def __init__(self, num_classes, e2e=True, classifier=False):
        super(OmniglotModel, self).__init__(e2e)

        self.num_classes = num_classes

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            # 2 x 2 - 64
        )

        if self.classifier or classifier:
            # 2 x 2 x 64 = 256
            self.classifier = MyLinear(256, self.num_classes)
        else:
            self.classifier = None

    def representation(self, x):
        shape = x.shape
        assert shape[-3:] == (1, 28, 28)
        x = x.view(-1, 1, 28, 28)

        x = self.conv(x)
        x = x.view(len(x), -1)
        return x.view(*shape[:-3], x.shape[-1])



class OmniglotModelFC(MetaLearningModel):
    def __init__(self, num_classes, e2e=True, classifier=False, net_width=1.):
        super(OmniglotModelFC, self).__init__(e2e)

        self.num_classes = num_classes
        self.nw = int(net_width)

        self.hidden = nn.Sequential(
            torch.nn.Linear(1*28*28, self.nw*64), #256),
            torch.nn.BatchNorm1d(self.nw*64),#256),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),#256, self.nw*128),
            torch.nn.BatchNorm1d(self.nw*64),#128),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),#128, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU()
        )


        if self.classifier or classifier:
            self.classifier = MyLinear(self.nw*64, self.num_classes)
        else:
            self.classifier = None
    def representation(self, x):
        shape = x.shape
        assert shape[-3:] == (1, 28, 28)
        x = x.view(-1, 1*28*28)
        x = self.hidden(x)
        #x = torch.nn.Linear(64, self.num_classes)
        return x.view(*shape[:-3], x.shape[-1])



class MiniImagenetModel(MetaLearningModel):
    def __init__(self, num_classes, e2e=True, classifier=False):
        super(MiniImagenetModel, self).__init__(e2e)

        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            # 84 x 84 - 3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )


        self.conv2 = nn.Sequential(
            # 42 x 42 - 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            # 21 x 21 - 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            # 11 x 11 - 32
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()

            # 6 x 6 - 32
        )

        self.conv = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        
        if self.classifier or classifier:
            # 6 x 6 x 32 = 1152
            self.classifier = MyLinear(1152, self.num_classes)
        else:
            self.classifier = None

    def representation(self, x):
        shape = x.shape
        assert shape[-3:] == (3, 84, 84)
        x = x.view(-1, 3, 84, 84)

        x = self.conv(x)
        x = x.view(len(x), -1)
        return x.view(*shape[:-3], x.shape[-1])



class MiniImagenetModelWidth(MetaLearningModel):
    def __init__(self, num_classes, e2e=True, classifier=False, net_width = 1.0):
        super(MiniImagenetModelWidth, self).__init__(e2e)

        self.num_classes = num_classes
        self.nw = net_width
        self.conv1 = nn.Sequential(
            # 84 x 84 - 3
            nn.Conv2d(3, int(32*self.nw), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d( int(32*self.nw)) ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )


        self.conv2 = nn.Sequential(
            # 42 x 42 - 32
            nn.Conv2d( int(32*self.nw) , int(64*self.nw) , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d( int(64*self.nw)) ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            # 21 x 21 - 32
            nn.Conv2d( int(64*self.nw) , int(128*self.nw), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d( int(128*self.nw)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            # 11 x 11 - 32
            nn.Conv2d( int(128*self.nw), int(128*self.nw), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d( int(128*self.nw)),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()

            # 6 x 6 - 32
        )

        self.conv = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)

        if self.classifier or classifier:
            # 6 x 6 x 32 = 1152
            self.classifier = MyLinear(int(6*6*128*self.nw), self.num_classes)
        else:
            self.classifier = None

    def representation(self, x):
        shape = x.shape
        assert shape[-3:] == (3, 84, 84)
        x = x.view(-1, 3, 84, 84)

        x = self.conv(x)
        #print(np.shape(x))
        x = x.view(len(x), -1)
        return x.view(*shape[:-3], x.shape[-1])



class MiniModelFC(MetaLearningModel):
    def __init__(self, num_classes, e2e=True, classifier=False, net_width=1.):
        super(MiniModelFC, self).__init__(e2e)

        self.num_classes = num_classes
        self.nw = int(net_width)

        self.hidden = nn.Sequential(
            torch.nn.Linear(3*84*84, self.nw*64), #256),
            torch.nn.BatchNorm1d(self.nw*64),#256),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),#256, self.nw*128),
            torch.nn.BatchNorm1d(self.nw*64),#128),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),#128, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nw*64, self.nw*64),
            torch.nn.BatchNorm1d(self.nw*64),
            torch.nn.ReLU()
        )


        if self.classifier or classifier:
            self.classifier = MyLinear(self.nw*64, self.num_classes)
        else:
            self.classifier = None

    def representation(self, x):
        shape = x.shape
        assert shape[-3:] == (3, 84, 84)
        x = x.view(-1, 3*84*84)
        #assert shape[-3:] == (1, 28, 28)
        #x = x.view(-1, 1*28*28)
        x = self.hidden(x)
        #x = torch.nn.Linear(64, self.num_classes)
        return x.view(*shape[:-3], x.shape[-1])

