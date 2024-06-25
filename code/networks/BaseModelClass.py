"""
Implementation of BaseModel taken and modified from here
https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/basemodel/basemodel.py
"""

import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    x = x.type(torch.float32)
    return Variable(x, requires_grad=requires_grad)

class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def params(self):
        for name, param in self.named_params(self):
            yield param
    def named_leaves(self):
        return []
    def named_submodules(self):
        return []
    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)
    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


    def __init__(self):
        super().__init__()
        self.best_loss = 0

    @abstractmethod
    def forward(self, x):
        pass

    #@abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        # TODO return optimizer?????
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        x = ckpt_dict['epoch']
        return ckpt_dict['model_state_dict']


    def save_checkpoint(self,
                        directory,
                        epoch, loss,
                        optimizer=None,
                        name=None,model_num=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.
            model_num(str):model1 or model2

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
                self.state_dict(),
            'optimizer_state_dict':
                optimizer.state_dict() if optimizer is not None else None,
            'epoch':
                epoch
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_last_epoch.pth".format(
                model_num,
                os.path.basename(directory),  # netD or netG
                )
                #'last')

        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss < loss:
            self.best_loss = loss
            name = "{}_{}_BEST.pth".format(
                model_num,
                os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()