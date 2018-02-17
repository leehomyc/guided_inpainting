from datetime import datetime
import os
import torch


# noinspection PyAttributeOutsideInit,PyShadowingBuiltins
class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        if self.opt.isTrain is True:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.save_dir = os.path.join(self.save_dir, current_time)
        else:
            self.save_dir = opt.checkpoints_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no back-propagation
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        print(save_path)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise Exception('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                       if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print(
                        'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)  # noqa 501
                except:
                    print(
                        'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)  # noqa 501
                    from sets import Set
                    not_initialized = Set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != \
                                pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate(self):
        pass
