import torch
from pydoc import locate

__torchvisionmodels__ = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]


def weights_init_all(m):
    if type(m) == 'Conv2d':
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == 'Linear':
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_pretrained(m):
    if type(m) == 'Linear':
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def linear_layer(in_features=1000, out_features=1, activation=None):
    if activation == 'sigmoid' or activation == 'Sigmoid':
        dense = torch.nn.Sequential(torch.nn.ReLU(),
                                    torch.nn.Dropout(p=0.5),
                                    torch.nn.Linear(in_features=in_features, out_features=out_features),
                                    torch.nn.Sigmoid())
    elif activation == 'softmax' or activation == 'Softmax':
        dense = torch.nn.Sequential(torch.nn.ReLU(),
                                    torch.nn.Dropout(p=0.5),
                                    torch.nn.Linear(in_features=in_features, out_features=out_features),
                                    torch.nn.Softmax(dim=1))
    else:
        dense = torch.nn.Sequential(torch.nn.ReLU(),
                                    torch.nn.Dropout(p=0.5),
                                    torch.nn.Linear(in_features=in_features, out_features=out_features))
    return dense


class Descriptor(torch.nn.Module):
    def __init__(self,
                 encoder_name='densenet169',
                 pretrain=True,
                 add_dense_layer=False,
                 activation=None,
                 num_input=1000,
                 num_output=1):
        super(Descriptor, self).__init__()
        """
        Classification or regression model based on one of many backbone architectures.
        encoder_name - backbone architecture, can use everything from pretrainedmodels and torchvision.models packages
        pretrain - should encoder use pretrained weights (imagenet) or not
        add_dense_layer - if True, adds linear layer (ReLU, Dropout, Linear) to the encode. If False, changes last layer
         of encoder to output num_output number of features.
        activation - sigmoid, softmax or None
        num_input - number of inputs to the last linear layer (usually 1000 by default)
        num_output - number of outputs.

        Example:
        from dp_tools.models import Descriptor
        model  = Descriptor(encoder_name='resnet34',
                            pretrain=True,
                            num_output=2,
                            add_dense_layer=False, 
                            activation='softmax')
        """

        dense = None
        self.add_dense_layer = add_dense_layer
        self.activation = activation
        if encoder_name in __torchvisionmodels__:
            encoder_class = 'torchvision.models.' + encoder_name
            encoder_hyperparams = {'pretrained': pretrain}
        else:
            encoder_class = 'pretrainedmodels.' + encoder_name
            if pretrain:
                encoder_hyperparams = {'pretrained': 'imagenet'}
                if encoder_name == 'pnasnet5large':
                    encoder_hyperparams['num_classes'] = 1000
            else:
                encoder_hyperparams = {'pretrained': None}
        encoder = locate(encoder_class)(**encoder_hyperparams)
        if add_dense_layer:
            dense = linear_layer(num_input, num_output, activation)
        else:
            if list(encoder.named_children())[-1][0] == 'last_linear':
                encoder.last_linear = linear_layer(encoder.last_linear.in_features, num_output, activation)
            else:
                encoder.fc = linear_layer(encoder.fc.in_features, num_output, activation)
        self.encoder = encoder
        self.dense = dense

    def forward(self, image):
        if self.add_dense_layer:
            x = self.encoder(image)
            x = self.dense(x)
        else:
            x = self.encoder(image)
        return x
