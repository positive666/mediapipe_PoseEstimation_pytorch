import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IrisBlock(nn.Module):
    """ Building block for mediapipe iris landmark model

    COnv + PRelu + DepthwiseConv + Conv + PRelu
    downsampling + channel padding for few blocks(when stride=2)
    channel padding values - 

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):
        super(IrisBlock, self).__init__()
        
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        
        padding = (kernel_size - 1) // 2

        self.conv_prelu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels/2), kernel_size=stride, stride=stride, padding=0, bias=True),
            nn.PReLU(int(out_channels/2))
        )
        self.depthwiseconv_conv = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=int(out_channels/2), 
                      kernel_size=kernel_size, stride=1, padding=padding, 
                      groups=int(out_channels/2), bias=True),
            nn.Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Downsample
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.prelu = nn.PReLU(out_channels)


    @torch.no_grad()
    def forward(self, x):
        """[summary]

        Args:
            x ([torch.Tensor]): [input tensor]

        Returns:
            [torch.Tensor]: [featues]
        """
        out = self.conv_prelu(x)
        out = self.depthwiseconv_conv(out)

        if self.stride == 2:
            x = self.max_pool(x)        

            if self.channel_pad > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.prelu(out + x)

class IrisLM(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self):
        """[summary]
        """
        super(IrisLM, self).__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True),
            nn.PReLU(64),

            IrisBlock(64, 64), IrisBlock(64, 64),
            IrisBlock(64, 64), IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),

            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2)
        )

        # iris_contour head
        self.iris_contour = nn.Sequential(
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True)
        )

        # eye_contour head
        self.eye_contour = nn.Sequential(
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128), IrisBlock(128, 128),
            nn.Conv2d(in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True)
        )


    @torch.no_grad()
    def forward(self, x):
        """ forward prop

        Args:
            x ([torch.Tensor]): [input Tensor]

        Returns:
            [list]: [eye_contour, iris_contour]
            eye_contour (batch_size, 213)
            (71 points)
            (x, y, z)
            (x, y) corresponds to image pixel locations
            iris_contour (batch_size, 15)
            (5, 3) 5 points
        """
        with torch.no_grad():
            x = F.pad(x, [0, 1, 0, 1], "constant", 0)

            # (_, 128, 8, 8)
            features = self.backbone(x)            

            # (_, 213, 1, 1)  
            eye_contour = self.eye_contour(features)            

            # (_, 15, 1, 1)
            iris_contour = self.iris_contour(features) 
        # (batch_size, 213)  (batch_size, 15)
        return [eye_contour.view(x.shape[0], -1), iris_contour.reshape(x.shape[0], -1)]
        
    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.backbone[0].weight.device

    def predict(self, img):
        """ single image inference

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))
        
        return self.batch_predict(img.unsqueeze(0))


    def batch_predict(self, x):
        """ batch inference

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))
        assert x.shape[1] == 3
        assert x.shape[2] == 64
        assert x.shape[3] == 64
        x = x.to(self._device())
        
        x = x.float()/255.0
        eye_contour, iris_contour = self.forward(x)

        return eye_contour.view(x.shape[0], -1), iris_contour.view(x.shape[0], -1)


    def test(self):
        """ Sample Inference"""
        inp = torch.randn(1, 3, 64, 64)
        output = self(inp)
        print(output[0].shape, output[1].shape)
        
    
class GetKeysDict:
    def __init__(self):
        self.iris_landmark_dict = {
                                    'eye_contour.0.conv_prelu.0.weight':  'conv2d_21/Kernel',
                                    'eye_contour.0.conv_prelu.0.bias':    'conv2d_21/Bias',

                                    'eye_contour.0.conv_prelu.1.weight':   'p_re_lu_21/Alpha',
                                    'eye_contour.0.depthwiseconv_conv.0.weight':   'depthwise_conv2d_10/Kernel',
                                    'eye_contour.0.depthwiseconv_conv.0.bias':     'depthwise_conv2d_10/Bias',
                                    'eye_contour.0.depthwiseconv_conv.1.weight':   'conv2d_22/Kernel',
                                    'eye_contour.0.depthwiseconv_conv.1.bias':     'conv2d_22/Bias',
                                    'eye_contour.0.prelu.weight':       'p_re_lu_22/Alpha',
                                    'eye_contour.1.conv_prelu.0.weight':   'conv2d_23/Kernel',
                                    'eye_contour.1.conv_prelu.0.bias':     'conv2d_23/Bias',
                                    'eye_contour.1.conv_prelu.1.weight':   'p_re_lu_23/Alpha',
                                    'eye_contour.1.depthwiseconv_conv.0.weight':   'depthwise_conv2d_11/Kernel',
                                    'eye_contour.1.depthwiseconv_conv.0.bias':     'depthwise_conv2d_11/Bias',
                                    'eye_contour.1.depthwiseconv_conv.1.weight':   'conv2d_24/Kernel',
                                    'eye_contour.1.depthwiseconv_conv.1.bias':     'conv2d_24/Bias',
                                    'eye_contour.1.prelu.weight':       'p_re_lu_24/Alpha',
                                    'eye_contour.2.conv_prelu.0.weight':   'conv2d_25/Kernel',
                                    'eye_contour.2.conv_prelu.0.bias':     'conv2d_25/Bias',
                                    'eye_contour.2.conv_prelu.1.weight':   'p_re_lu_25/Alpha',
                                    'eye_contour.2.depthwiseconv_conv.0.weight':   'depthwise_conv2d_12/Kernel',
                                    'eye_contour.2.depthwiseconv_conv.0.bias':     'depthwise_conv2d_12/Bias',
                                    'eye_contour.2.depthwiseconv_conv.1.weight':   'conv2d_26/Kernel',
                                    'eye_contour.2.depthwiseconv_conv.1.bias':     'conv2d_26/Bias',
                                    'eye_contour.2.prelu.weight':       'p_re_lu_26/Alpha',
                                    'eye_contour.3.conv_prelu.0.weight':   'conv2d_27/Kernel',
                                    'eye_contour.3.conv_prelu.0.bias':     'conv2d_27/Bias',
                                    'eye_contour.3.conv_prelu.1.weight':   'p_re_lu_27/Alpha',
                                    'eye_contour.3.depthwiseconv_conv.0.weight':   'depthwise_conv2d_13/Kernel',
                                    'eye_contour.3.depthwiseconv_conv.0.bias':     'depthwise_conv2d_13/Bias',
                                    'eye_contour.3.depthwiseconv_conv.1.weight':   'conv2d_28/Kernel',
                                    'eye_contour.3.depthwiseconv_conv.1.bias':     'conv2d_28/Bias',
                                    'eye_contour.3.prelu.weight':       'p_re_lu_28/Alpha',
                                    'eye_contour.4.conv_prelu.0.weight':   'conv2d_29/Kernel',
                                    'eye_contour.4.conv_prelu.0.bias':     'conv2d_29/Bias',
                                    'eye_contour.4.conv_prelu.1.weight':   'p_re_lu_29/Alpha',
                                    'eye_contour.4.depthwiseconv_conv.0.weight':   'depthwise_conv2d_14/Kernel',
                                    'eye_contour.4.depthwiseconv_conv.0.bias':     'depthwise_conv2d_14/Bias',
                                    'eye_contour.4.depthwiseconv_conv.1.weight':   'conv2d_30/Kernel',
                                    'eye_contour.4.depthwiseconv_conv.1.bias':     'conv2d_30/Bias',
                                    'eye_contour.4.prelu.weight':       'p_re_lu_30/Alpha',
                                    'eye_contour.5.conv_prelu.0.weight':   'conv2d_31/Kernel',
                                    'eye_contour.5.conv_prelu.0.bias':     'conv2d_31/Bias',
                                    'eye_contour.5.conv_prelu.1.weight':   'p_re_lu_31/Alpha',
                                    'eye_contour.5.depthwiseconv_conv.0.weight':   'depthwise_conv2d_15/Kernel',
                                    'eye_contour.5.depthwiseconv_conv.0.bias':     'depthwise_conv2d_15/Bias',
                                    'eye_contour.5.depthwiseconv_conv.1.weight':   'conv2d_32/Kernel',
                                    'eye_contour.5.depthwiseconv_conv.1.bias':     'conv2d_32/Bias',
                                    'eye_contour.5.prelu.weight':       'p_re_lu_32/Alpha',
                                    'eye_contour.6.conv_prelu.0.weight':   'conv2d_33/Kernel',
                                    'eye_contour.6.conv_prelu.0.bias':     'conv2d_33/Bias',
                                    'eye_contour.6.conv_prelu.1.weight':   'p_re_lu_33/Alpha',
                                    'eye_contour.6.depthwiseconv_conv.0.weight':   'depthwise_conv2d_16/Kernel',
                                    'eye_contour.6.depthwiseconv_conv.0.bias':     'depthwise_conv2d_16/Bias',
                                    'eye_contour.6.depthwiseconv_conv.1.weight':   'conv2d_34/Kernel',
                                    'eye_contour.6.depthwiseconv_conv.1.bias':     'conv2d_34/Bias',
                                    'eye_contour.6.prelu.weight':       'p_re_lu_34/Alpha',
                                    'eye_contour.7.conv_prelu.0.weight':   'conv2d_35/Kernel',
                                    'eye_contour.7.conv_prelu.0.bias':     'conv2d_35/Bias',
                                    'eye_contour.7.conv_prelu.1.weight':   'p_re_lu_35/Alpha',
                                    'eye_contour.7.depthwiseconv_conv.0.weight':   'depthwise_conv2d_17/Kernel',
                                    'eye_contour.7.depthwiseconv_conv.0.bias':     'depthwise_conv2d_17/Bias',
                                    'eye_contour.7.depthwiseconv_conv.1.weight':   'conv2d_36/Kernel',
                                    'eye_contour.7.depthwiseconv_conv.1.bias':     'conv2d_36/Bias',
                                    'eye_contour.7.prelu.weight':       'p_re_lu_36/Alpha',
                                    'eye_contour.8.weight':           'conv_eyes_contours_and_brows/Kernel',
                                    'eye_contour.8.bias':             'conv_eyes_contours_and_brows/Bias',

                                    'iris_contour.0.conv_prelu.0.weight':  'conv2d_37/Kernel',
                                    'iris_contour.0.conv_prelu.0.bias':    'conv2d_37/Bias',
                                    'iris_contour.0.conv_prelu.1.weight':  'p_re_lu_37/Alpha',
                                    'iris_contour.0.depthwiseconv_conv.0.weight':  'depthwise_conv2d_18/Kernel',
                                    'iris_contour.0.depthwiseconv_conv.0.bias':    'depthwise_conv2d_18/Bias',
                                    'iris_contour.0.depthwiseconv_conv.1.weight':  'conv2d_38/Kernel',
                                    'iris_contour.0.depthwiseconv_conv.1.bias':    'conv2d_38/Bias',
                                    'iris_contour.0.prelu.weight':      'p_re_lu_38/Alpha',
                                    'iris_contour.1.conv_prelu.0.weight':  'conv2d_39/Kernel',
                                    'iris_contour.1.conv_prelu.0.bias':    'conv2d_39/Bias',
                                    'iris_contour.1.conv_prelu.1.weight':  'p_re_lu_39/Alpha',
                                    'iris_contour.1.depthwiseconv_conv.0.weight':  'depthwise_conv2d_19/Kernel',
                                    'iris_contour.1.depthwiseconv_conv.0.bias':    'depthwise_conv2d_19/Bias',
                                    'iris_contour.1.depthwiseconv_conv.1.weight':  'conv2d_40/Kernel',
                                    'iris_contour.1.depthwiseconv_conv.1.bias':    'conv2d_40/Bias',
                                    'iris_contour.1.prelu.weight':      'p_re_lu_40/Alpha',
                                    'iris_contour.2.conv_prelu.0.weight':  'conv2d_41/Kernel',
                                    'iris_contour.2.conv_prelu.0.bias':    'conv2d_41/Bias',
                                    'iris_contour.2.conv_prelu.1.weight':  'p_re_lu_41/Alpha',
                                    'iris_contour.2.depthwiseconv_conv.0.weight':  'depthwise_conv2d_20/Kernel',
                                    'iris_contour.2.depthwiseconv_conv.0.bias':    'depthwise_conv2d_20/Bias',
                                    'iris_contour.2.depthwiseconv_conv.1.weight':  'conv2d_42/Kernel',
                                    'iris_contour.2.depthwiseconv_conv.1.bias':    'conv2d_42/Bias',
                                    'iris_contour.2.prelu.weight':      'p_re_lu_42/Alpha',
                                    'iris_contour.3.conv_prelu.0.weight':  'conv2d_43/Kernel',
                                    'iris_contour.3.conv_prelu.0.bias':    'conv2d_43/Bias',
                                    'iris_contour.3.conv_prelu.1.weight':  'p_re_lu_43/Alpha',
                                    'iris_contour.3.depthwiseconv_conv.0.weight':  'depthwise_conv2d_21/Kernel',
                                    'iris_contour.3.depthwiseconv_conv.0.bias':    'depthwise_conv2d_21/Bias',
                                    'iris_contour.3.depthwiseconv_conv.1.weight':  'conv2d_44/Kernel',
                                    'iris_contour.3.depthwiseconv_conv.1.bias':    'conv2d_44/Bias',
                                    'iris_contour.3.prelu.weight':      'p_re_lu_44/Alpha',
                                    'iris_contour.4.conv_prelu.0.weight':  'conv2d_45/Kernel',
                                    'iris_contour.4.conv_prelu.0.bias':    'conv2d_45/Bias',
                                    'iris_contour.4.conv_prelu.1.weight':  'p_re_lu_45/Alpha',
                                    'iris_contour.4.depthwiseconv_conv.0.weight':  'depthwise_conv2d_22/Kernel',
                                    'iris_contour.4.depthwiseconv_conv.0.bias':    'depthwise_conv2d_22/Bias',
                                    'iris_contour.4.depthwiseconv_conv.1.weight':  'conv2d_46/Kernel',
                                    'iris_contour.4.depthwiseconv_conv.1.bias':    'conv2d_46/Bias',
                                    'iris_contour.4.prelu.weight':      'p_re_lu_46/Alpha',
                                    'iris_contour.5.conv_prelu.0.weight':  'conv2d_47/Kernel',
                                    'iris_contour.5.conv_prelu.0.bias':    'conv2d_47/Bias',
                                    'iris_contour.5.conv_prelu.1.weight':  'p_re_lu_47/Alpha',
                                    'iris_contour.5.depthwiseconv_conv.0.weight':  'depthwise_conv2d_23/Kernel',
                                    'iris_contour.5.depthwiseconv_conv.0.bias':    'depthwise_conv2d_23/Bias',
                                    'iris_contour.5.depthwiseconv_conv.1.weight':  'conv2d_48/Kernel',
                                    'iris_contour.5.depthwiseconv_conv.1.bias':    'conv2d_48/Bias',
                                    'iris_contour.5.prelu.weight':      'p_re_lu_48/Alpha',
                                    'iris_contour.6.conv_prelu.0.weight':  'conv2d_49/Kernel',
                                    'iris_contour.6.conv_prelu.0.bias':    'conv2d_49/Bias',
                                    'iris_contour.6.conv_prelu.1.weight':  'p_re_lu_49/Alpha',
                                    'iris_contour.6.depthwiseconv_conv.0.weight':  'depthwise_conv2d_24/Kernel',
                                    'iris_contour.6.depthwiseconv_conv.0.bias':    'depthwise_conv2d_24/Bias',
                                    'iris_contour.6.depthwiseconv_conv.1.weight':  'conv2d_50/Kernel',
                                    'iris_contour.6.depthwiseconv_conv.1.bias':    'conv2d_50/Bias',
                                    'iris_contour.6.prelu.weight':      'p_re_lu_50/Alpha',
                                    'iris_contour.7.conv_prelu.0.weight':  'conv2d_51/Kernel',
                                    'iris_contour.7.conv_prelu.0.bias':    'conv2d_51/Bias',
                                    'iris_contour.7.conv_prelu.1.weight':  'p_re_lu_51/Alpha',
                                    'iris_contour.7.depthwiseconv_conv.0.weight':  'depthwise_conv2d_25/Kernel',
                                    'iris_contour.7.depthwiseconv_conv.0.bias':    'depthwise_conv2d_25/Bias',
                                    'iris_contour.7.depthwiseconv_conv.1.weight':  'conv2d_52/Kernel',
                                    'iris_contour.7.depthwiseconv_conv.1.bias':    'conv2d_52/Bias',
                                    'iris_contour.7.prelu.weight':      'p_re_lu_52/Alpha',
                                    'iris_contour.8.weight':          'conv_iris/Kernel',
                                    'iris_contour.8.bias':            'conv_iris/Bias'
                                }